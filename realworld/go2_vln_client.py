import os
import io
import json
import time
import math
import shutil
import datetime
import threading
import requests
import numpy as np
import PIL.Image as PIL_Image
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from unitree_go.msg import SportModeState
from unitree_api.msg import Request
from unitree_api.msg import RequestHeader

from utils import ReadWriteLock
from pid_controller import PID_controller


@dataclass
class ControlConfig:
    """Go2 low-level motion control and PID guard parameters."""
    pos_tol: float = 0.06
    yaw_tol: float = math.radians(5.0)
    yaw_turn_enter: float = math.radians(7.0)
    yaw_turn_exit: float = math.radians(3.0)
    min_w: float = 0.25

    yaw_stuck_trigger_count: int = 10
    yaw_stuck_delta: float = math.radians(0.3)
    stuck_boost_w: float = 0.65

    fwd_tol: float = 0.12
    fwd_brake_dist: float = 0.10
    fwd_cruise_v: float = 0.30
    fwd_slow_v: float = 0.22
    fwd_heading_kp: float = 1.8
    fwd_heading_deadband: float = math.radians(1.5)
    fwd_heading_gate: float = math.radians(10.0)
    fwd_max_w: float = 0.35

    fwd_stuck_trigger_count: int = 12
    fwd_stuck_delta: float = 0.003
    stuck_boost_v: float = 0.40

    kp_lateral: float = 1.0
    max_vy: float = 0.18
    cmd_vy_sign: float = 1.0


def clamp_min_magnitude(cmd, min_abs, eps=1e-6):
    if abs(cmd) < eps:
        return 0.0
    if abs(cmd) < min_abs:
        return float(np.sign(cmd) * min_abs)
    return cmd


class Go2VlnPipeline(Node):
    def __init__(self):
        super().__init__('go2_vln_pipeline')

        self.cv_bridge = CvBridge()

        self.rgb_rw_lock = ReadWriteLock()
        self.odom_rw_lock = ReadWriteLock()

        self.rgb_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        self.request_cnt = 0
        self.odom_cnt = 0

        self.should_plan = False
        self.current_action = None
        self.action_queue = []
        self.goal_queue = []
        self.goal_source = "idle"
        self.policy_init = True
        self.current_instruction = ""

        self.cfg = ControlConfig()
        self.pid = PID_controller(
            Kp_trans=2.4,
            Kd_trans=0.4,
            Kp_yaw=3.0,
            Kd_yaw=0.5,
            max_v=0.6,
            max_w=1.2,
        )

        self.rgb_sub = self.create_subscription(Image, "/camera/color/image_raw", self.rgb_callback, 1)
        self.odom_sub = self.create_subscription(SportModeState, "/sportmodestate", self.odom_callback, 10)
        self.control_pub = self.create_publisher(Request, '/api/sport/request', 5)

        self.dataset_dir = "../vln_dataset"
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.current_episode_dir = os.path.join(self.dataset_dir, "tmp_episode")

    # ==========================
    # ROS2 callbacks
    # ==========================
    def rgb_callback(self, msg):
        self.rgb_rw_lock.acquire_write()
        self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')[:, :, :]
        self.rgb_rw_lock.release_write()

    def odom_callback(self, msg):
        DOWNSAMPLE_RATIO = 5
        self.odom_cnt += 1
        if self.odom_cnt % DOWNSAMPLE_RATIO != 0:
            return

        self.odom_rw_lock.acquire_write()
        yaw = msg.imu_state.rpy[2]
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)],
                       [np.sin(yaw),  np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.position[0], msg.position[1]]
        self.vel = [msg.velocity[0], msg.yaw_speed]

        if self.odom_cnt == DOWNSAMPLE_RATIO:
            self.homo_goal = self.homo_odom.copy()

        self.odom_rw_lock.release_write()

    # ==========================
    # State management and goals
    # ==========================
    def trigger_replan(self):
        self.should_plan = True

    def sync_goal(self):
        if self.homo_odom is not None:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_odom is None:
            print("Warning: Odometry not received yet. Cannot initialize goal.")
            return

        self.sync_goal()
        self.homo_goal = self._apply_actions_to_goal(self.homo_goal.copy(), actions)

    def _apply_actions_to_goal(self, homo_goal, actions):
        if homo_goal is None:
            return None

        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action in (2, 3):
                angle_deg = 15.0 if each_action == 2 else -15.0
                angle = math.radians(angle_deg)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1],
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])

        return homo_goal

    def prepare_action_goals(self, actions):
        """Execute one step while preserving precomputed goals for a requested sequence."""
        if self.homo_odom is None:
            print("Warning: Odometry not received yet. Cannot initialize goal.")
            self.action_queue = []
            self.goal_queue = []
            self.current_action = None
            self.goal_source = "no_odom"
            return False

        if not isinstance(actions, list):
            actions = [actions]

        if not actions:
            self.action_queue = []
            self.goal_queue = []
            self.current_action = None
            self.goal_source = "empty"
            return False

        requested_action = actions[0]
        if requested_action == 0:
            self.action_queue = []
            self.goal_queue = []
            self.sync_goal()
            self.current_action = 0
            self.goal_source = "stop"
            return True

        if self.action_queue and self.goal_queue:
            if requested_action == self.action_queue[0]:
                self.current_action = self.action_queue.pop(0)
                self.homo_goal = self.goal_queue.pop(0)
                self.goal_source = "precomputed_sequence"
                if len(actions) > 1:
                    planned_goal = self.homo_goal.copy()
                    self.action_queue = []
                    self.goal_queue = []
                    for action in actions[1:]:
                        planned_goal = self._apply_actions_to_goal(planned_goal.copy(), [action])
                        self.action_queue.append(action)
                        self.goal_queue.append(planned_goal.copy())
                return True

            print(
                f"--> Pending precomputed sequence starts with {self.action_queue[0]}, "
                f"but requested {requested_action}; replanning from current odom."
            )
            self.action_queue = []
            self.goal_queue = []

        self.sync_goal()
        planned_goal = self.homo_goal.copy()
        planned_steps = []
        for action in actions:
            planned_goal = self._apply_actions_to_goal(planned_goal.copy(), [action])
            planned_steps.append((action, planned_goal.copy()))

        if not planned_steps:
            self.action_queue = []
            self.goal_queue = []
            self.current_action = None
            self.goal_source = "empty"
            return False

        first_action, first_goal = planned_steps[0]
        self.current_action = first_action
        self.homo_goal = first_goal
        self.action_queue = [action for action, _ in planned_steps[1:]]
        self.goal_queue = [goal for _, goal in planned_steps[1:]]
        self.goal_source = "sync" if len(planned_steps) == 1 else "sync_sequence"
        return True

    def move(self, vx, vy, vyaw):
        SPORT_API_ID_MOVE = 1008
        parameter = json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})
        header = RequestHeader()
        header.identity._api_id = SPORT_API_ID_MOVE
        header.identity.id = time.monotonic_ns()
        request = Request(parameter=parameter, header=header)
        self.control_pub.publish(request)

    # ==========================
    # VLN inference and logging
    # ==========================
    def eval_vln(self, image, instruction, url='http://10.1.125.11:5801/eval_vln', debug=False):
        if debug:
            self.policy_init = False
            return np.random.choice([0, 1, 2, 3], size=(1,)).tolist(), True, 0, "debug_mode"

        image_pil = PIL_Image.fromarray(image)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format='jpeg')
        image_bytes.seek(0)

        data = {"reset": self.policy_init, "instruction": instruction}
        json_data = json.dumps(data)

        files = {'image': ('rgb_image', image_bytes, 'image/jpg')}
        start = time.time()
        response = requests.post(url, files=files, data={'json': json_data}, timeout=150)
        response.raise_for_status()
        self.policy_init = False

        print(f"\nTotal time (delay + policy): {time.time() - start:.3f}s")
        print(response.text)

        resp_data = json.loads(response.text)
        model_actions = resp_data.get('action_sequence', resp_data['action'])
        return (
            model_actions,
            resp_data.get('is_inference', False),
            resp_data.get('pending_count', 0),
            resp_data.get('raw_model_output', ''),
        )

    def get_human_verification(self, model_actions):
        action_map = {0: "Stop", 1: "Move FWD 25cm", 2: "Turn L 15deg", 3: "Turn R 15deg"}
        key_to_action = {"w": 1, "a": 2, "d": 3}
        max_manual_repeat = 3

        if isinstance(model_actions, list) and len(model_actions) > 1:
            readable = [action_map.get(a, str(a)) for a in model_actions]
            print(f"Model Proposed Sequence: {readable} ({model_actions})")
        else:
            model_action = model_actions[0] if isinstance(model_actions, list) else model_actions
            print(f"Qwen-VL Proposal: {action_map.get(model_action, 'Unknown')} ({model_action})")

        print("Commands: [ENTER]=Accept | w/a/d/s=Override | w3/www=Precompute repeat goals, execute one step | q=Discard | r=End Episode & Reset")

        while rclpy.ok():
            val = input("Your choice: ").strip().lower()
            manual_actions = None
            if val and val[0] in key_to_action:
                if val in key_to_action:
                    repeat_count = 1
                elif val[1:].isdigit():
                    repeat_count = int(val[1:])
                elif all(ch == val[0] for ch in val):
                    repeat_count = len(val)
                else:
                    repeat_count = 0

                if 1 <= repeat_count <= max_manual_repeat:
                    manual_actions = [key_to_action[val[0]]] * repeat_count

            if val == "":
                return model_actions if isinstance(model_actions, list) else [model_actions]
            if manual_actions is not None:
                return manual_actions
            if val == "s":
                return [0]
            if val == "q":
                print("--> Discarding frame.")
                return None
            if val == "r":
                print("--> Ending current episode. Preparing for next instruction.")
                return "RESET"

            print("Invalid input.")

        return None

    def _prompt_save_episode(self):
        if os.path.exists(self.current_episode_dir) and os.listdir(self.current_episode_dir):
            save_choice = input("Episode finished. Save this episode? [y/n]: ").strip().lower()
            if save_choice == 'y':
                new_episode_name = f"episode_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                final_episode_dir = os.path.join(self.dataset_dir, new_episode_name)
                os.rename(self.current_episode_dir, final_episode_dir)
                print(f"--> Episode saved to {final_episode_dir}")
            else:
                shutil.rmtree(self.current_episode_dir)
                print("--> Episode discarded.")

        if os.path.exists(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
        os.makedirs(self.current_episode_dir, exist_ok=True)

    def _log_step(self, image_rgb, final_actions, model_action_list, is_inference, pending_count, raw_model_output, inf_step, exec_slot, requested_actions=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpeg"

        image_pil = PIL_Image.fromarray(image_rgb)
        image_pil.save(os.path.join(self.current_episode_dir, filename))

        self.odom_rw_lock.acquire_read()
        cur_x = self.homo_odom[0, 3] if self.homo_odom is not None else None
        cur_y = self.homo_odom[1, 3] if self.homo_odom is not None else None
        cur_yaw = math.atan2(self.homo_odom[1, 0], self.homo_odom[0, 0]) if self.homo_odom is not None else None

        goal_x = self.homo_goal[0, 3] if self.homo_goal is not None else None
        goal_y = self.homo_goal[1, 3] if self.homo_goal is not None else None
        goal_yaw = math.atan2(self.homo_goal[1, 0], self.homo_goal[0, 0]) if self.homo_goal is not None else None
        goal_source = self.goal_source
        remaining_action_queue = list(self.action_queue)
        self.odom_rw_lock.release_read()

        log_entry = {
            "timestamp": timestamp,
            "image_filename": filename,
            "instruction": self.current_instruction,
            "inference_step": inf_step,
            "inference_action_slot": exec_slot,
            "is_inference": is_inference,
            "pending_count": pending_count,
            "raw_model_output": raw_model_output,
            "model_action_raw": model_action_list,
            "model_action_count": len(model_action_list),
            "requested_actions": requested_actions if requested_actions is not None else final_actions,
            "executed_actions": final_actions,
            "goal_source": goal_source,
            "remaining_precomputed_actions": remaining_action_queue,
            "step_position": {
                "x": cur_x,
                "y": cur_y,
                "yaw": cur_yaw,
                "yaw_deg": math.degrees(cur_yaw) if cur_yaw is not None else None,
            },
            "step_goal": {
                "x": goal_x,
                "y": goal_y,
                "yaw": goal_yaw,
                "yaw_deg": math.degrees(goal_yaw) if goal_yaw is not None else None,
            },
        }

        log_file = os.path.join(self.current_episode_dir, "trajectory_log.txt")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # ==========================
    # Threads
    # ==========================
    def planning_thread(self):
        inference_step_counter = 0
        action_execution_count = 0
        episode_terminated = False

        while rclpy.ok():
            if not self.should_plan:
                time.sleep(0.01)
                continue

            if self.policy_init:
                self._prompt_save_episode()
                print("\n" + "=" * 50)
                self.current_instruction = input("Enter new navigation instruction: ").strip()
                print("=" * 50 + "\n")

                inference_step_counter = 0
                action_execution_count = 0
                episode_terminated = False

            self.rgb_rw_lock.acquire_read()
            image = self.rgb_image
            self.rgb_rw_lock.release_read()

            if image is None:
                time.sleep(0.1)
                continue

            model_action_list, is_inf, pending_cnt, raw_out = self.eval_vln(image[..., ::-1], self.current_instruction)
            final_actions = self.get_human_verification(model_action_list)

            if final_actions == "RESET":
                self.policy_init = True
                self.should_plan = False
                self.odom_rw_lock.acquire_write()
                self.sync_goal()
                self.current_action = None
                self.action_queue = []
                self.goal_queue = []
                self.goal_source = "idle"
                self.odom_rw_lock.release_write()
                continue

            if final_actions is not None:
                requested_actions = final_actions if isinstance(final_actions, list) else [final_actions]
                self.odom_rw_lock.acquire_write()
                self.request_cnt += 1
                prepared = self.prepare_action_goals(requested_actions)
                executed_actions = [self.current_action] if prepared else []
                self.odom_rw_lock.release_write()

                if prepared:
                    inference_step_counter += 1
                    self._log_step(
                        image[..., ::-1],
                        executed_actions,
                        model_action_list,
                        is_inf,
                        pending_cnt,
                        raw_out,
                        inference_step_counter,
                        action_execution_count,
                        requested_actions=requested_actions,
                    )
                    action_execution_count += 1

                if 0 in executed_actions:
                    episode_terminated = True

                self.should_plan = False
            else:
                self.should_plan = False

            if episode_terminated and not self.should_plan:
                while not self.should_plan and rclpy.ok():
                    time.sleep(0.1)

            if episode_terminated and not self.policy_init and self.should_plan:
                self._prompt_save_episode()
                inference_step_counter = 0
                action_execution_count = 0
                episode_terminated = False
                self.policy_init = True

            time.sleep(0.1)

    def control_thread(self):
        c = self.cfg
        prev_abs_e_r, yaw_stuck_count = None, 0
        prev_e_fwd, fwd_stuck_count = None, 0
        turn_only_mode = False

        while rclpy.ok():
            if self.should_plan:
                self.move(0.0, 0.0, 0.0)
                time.sleep(0.033)
                continue

            self.odom_rw_lock.acquire_read()
            homo_odom = self.homo_odom.copy() if self.homo_odom is not None else None
            homo_goal = self.homo_goal.copy() if self.homo_goal is not None else None
            vel = self.vel.copy() if self.vel is not None else None
            action = self.current_action
            goal_source = self.goal_source
            self.odom_rw_lock.release_read()

            if homo_odom is None or vel is None or homo_goal is None:
                time.sleep(0.033)
                continue

            v_pid, w_pid, e_x, e_y, e_r = self.pid.solve(homo_odom, homo_goal, vel)

            dx = homo_goal[0, 3] - homo_odom[0, 3]
            dy = homo_goal[1, 3] - homo_odom[1, 3]
            goal_yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
            e_fwd = dx * math.cos(goal_yaw) + dy * math.sin(goal_yaw)
            abs_er = abs(e_r)

            v, vy, w = v_pid, 0.0, w_pid

            if action == 1:
                if abs_er > c.fwd_heading_gate:
                    v, vy = 0.0, 0.0
                    w = float(np.sign(e_r) * max(c.min_w, min(c.fwd_max_w, abs(c.fwd_heading_kp * e_r))))
                else:
                    v = c.fwd_cruise_v if e_fwd > c.fwd_brake_dist else c.fwd_slow_v

                    if prev_e_fwd is not None and abs(prev_e_fwd - e_fwd) < c.fwd_stuck_delta:
                        fwd_stuck_count += 1
                    else:
                        fwd_stuck_count = 0

                    if fwd_stuck_count >= c.fwd_stuck_trigger_count:
                        v = c.stuck_boost_v
                    prev_e_fwd = e_fwd

                    vy = 0.0
                    w = 0.0 if abs_er < c.fwd_heading_deadband else float(np.clip(c.fwd_heading_kp * e_r, -c.fwd_max_w, c.fwd_max_w))
            else:
                prev_e_fwd, fwd_stuck_count = None, 0

                if turn_only_mode and abs_er < c.yaw_turn_exit:
                    turn_only_mode = False
                elif not turn_only_mode and abs_er > c.yaw_turn_enter:
                    turn_only_mode = True

                if turn_only_mode:
                    v, vy = 0.0, 0.0
                else:
                    vy = c.cmd_vy_sign * float(np.clip(c.kp_lateral * e_y, -c.max_vy, c.max_vy))

                dist = math.sqrt(e_x ** 2 + e_y ** 2)
                if dist > c.pos_tol:
                    speed = math.sqrt(v ** 2 + vy ** 2)
                    if speed < 0.12:
                        scale = 0.12 / (speed + 1e-6)
                        v *= scale
                        vy *= scale

                if abs_er > c.yaw_tol:
                    w = clamp_min_magnitude(w, c.min_w)

            if abs_er > c.yaw_tol:
                if prev_abs_e_r is not None and abs(prev_abs_e_r - abs_er) < c.yaw_stuck_delta:
                    yaw_stuck_count += 1
                else:
                    yaw_stuck_count = 0

                if yaw_stuck_count >= c.yaw_stuck_trigger_count:
                    w = float(np.sign(e_r) * c.stuck_boost_w)
                    yaw_stuck_count = 0
                prev_abs_e_r = abs_er
            else:
                prev_abs_e_r, yaw_stuck_count = None, 0

            self.move(v, vy, w)
            odom_yaw = math.atan2(homo_odom[1, 0], homo_odom[0, 0])

            if action == 1:
                print(
                    f"FWD[{goal_source}]: action={action}, "
                    f"odom_yaw={math.degrees(odom_yaw):.1f}, goal_yaw={math.degrees(goal_yaw):.1f}, "
                    f"e_fwd={e_fwd:.4f}, e_y={e_y:.4f}, e_r={e_r:.4f}, "
                    f"v={v:.3f}, vy={vy:.3f}, w={w:.3f}"
                )
            else:
                print(
                    f"MOVE[{goal_source}]: action={action}, "
                    f"odom_yaw={math.degrees(odom_yaw):.1f}, goal_yaw={math.degrees(goal_yaw):.1f}, "
                    f"dist={math.sqrt(e_x ** 2 + e_y ** 2):.4f}, e_y={e_y:.4f}, e_r={e_r:.4f}, "
                    f"v={v:.3f}, vy={vy:.3f}, w={w:.3f}"
                )

            if action == 1:
                done = e_fwd < c.fwd_tol and abs(e_r) < math.radians(10.0)
            elif action in (2, 3):
                done = abs(e_r) < math.radians(8.0)
            else:
                done = math.sqrt(e_x ** 2 + e_y ** 2) < 0.10 and abs(e_r) < math.radians(8.0)

            if done:
                self.move(0.0, 0.0, 0.0)
                prev_e_fwd, fwd_stuck_count = None, 0

                self.odom_rw_lock.acquire_write()
                self.current_action = None
                self.goal_source = "idle"
                self.trigger_replan()
                self.odom_rw_lock.release_write()

            time.sleep(0.033)


if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot.")

    rclpy.init()
    pipeline = Go2VlnPipeline()

    control_thread_instance = threading.Thread(target=pipeline.control_thread, daemon=True)
    planning_thread_instance = threading.Thread(target=pipeline.planning_thread, daemon=True)

    control_thread_instance.start()
    planning_thread_instance.start()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        try:
            pipeline.move(0.0, 0.0, 0.0)
        except Exception:
            pass
        pipeline.destroy_node()
        rclpy.shutdown()
