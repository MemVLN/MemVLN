import numpy as np
import torch


def calc_relative_pos_and_yaw(actions, turn_unit=15):
    relative_pos_and_yaw = []
    curr_x = 0.0
    curr_y = 0.0
    curr_yaw = 0.0
    relative_pos_and_yaw.append((curr_x, curr_y, curr_yaw))
    for action in actions:
        if action == "turn left":
            curr_yaw -= turn_unit
            if curr_yaw < -180:
                curr_yaw += 360
        elif action == "turn right":
            curr_yaw += turn_unit
            if curr_yaw > 180:
                curr_yaw -= 360
        elif action == "move forward":
            curr_x += np.sin(np.deg2rad(curr_yaw)) * 0.25
            curr_y += np.cos(np.deg2rad(curr_yaw)) * 0.25
        elif action == "stop":
            continue
        curr_x = round(curr_x, 3)
        curr_y = round(curr_y, 3)
        # curr_yaw = (np.deg2rad(curr_yaw) + np.pi) % (2 * np.pi) - np.pi
        # assert curr_yaw >= -np.pi and curr_yaw <= np.pi
        # relative_pos_and_yaw.append((curr_x, curr_y, np.sin(curr_yaw), np.cos*curr_yaw))
        relative_pos_and_yaw.append((curr_x, curr_y, curr_yaw))
    relative_pos_and_yaw = np.array(relative_pos_and_yaw)

    return relative_pos_and_yaw


def merge_actions(actions, merge_move=True, merge_turn=True, move_unit=25, turn_unit=15):
    if len(actions) == 0:
        return [], []
    prev = actions[0]
    csd_idx_list = []
    total_count = 0
    count = 1
    merged_actions = []

    def flush_action(action, count):
        if action == "stop":
            merged_actions.append(action)
        elif action == "move forward":
            merged_actions.append(action + f" {count * move_unit} cm")
        else:
            merged_actions.append(action + f" {count * turn_unit} degree")

    for action in actions[1:]:
        if action == "move forward" and merge_move and action == prev and count < 3:
            count += 1
        elif "turn" in action and merge_turn and action == prev and count < 3:
            count += 1
        else:
            csd_idx_list.append(total_count)
            total_count += count
            flush_action(prev, count)
            count = 1
            prev = action

    csd_idx_list.append(total_count)
    flush_action(prev, count)

    assert len(csd_idx_list) == len(merged_actions), "Each merged action should have an index corresponding to original actions"
    return merged_actions, csd_idx_list


def pad_tensor(tensor, dim=0, max_len=None, pad=0, side='right'):
    assert tensor.shape[dim] <= max_len
    if tensor.shape[dim] == max_len:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = max_len - shape[dim]
    res = torch.ones(shape, dtype=tensor.dtype, device=tensor.device) * pad
    if side == 'right':
        res = torch.cat([tensor, res], dim=dim)
    else:
        res = torch.cat([res, tensor], dim=dim)
    return res


def pad_tensors(tensor_list, dim=0, max_len=None, pad=0, side='right', return_mask=False):
    lens = [x.shape[dim] for x in tensor_list]
    if max_len is None:
        max_len = max(lens)

    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    padded_tensors = []
    for tensor in tensor_list:
        padded_tensors.append(pad_tensor(tensor, dim=dim, max_len=max_len, pad=pad, side=side))
    padded_tensors = torch.stack(padded_tensors, dim=0).to(dtype).to(device)

    if return_mask:
        mask = torch.arange(max_len).to(device)[None, :] < torch.LongTensor(lens).to(device)[:, None]   # True as valid
        if side == 'left':
            mask = torch.flip(mask, dims=[1])
        return padded_tensors, mask
    else:
        return padded_tensors


def dict_to_cuda(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device) for ele in v]
        # elif (
        #     isinstance(input_dict[k], list)
        #     and len(input_dict[k]) > 0
        #     and isinstance(input_dict[k][0], Det3DDataElement)
        # ):
        #     input_dict[k] = [ele.to(device) for ele in v]

    return input_dict
