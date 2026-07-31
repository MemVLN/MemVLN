import os
import socket
import subprocess
from pathlib import Path

import submitit

_SUBMITIT_PORT_BASE = 15000
_SUBMITIT_PORT_COUNT = 20000


def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        # getsockname returns the address and port the socket is bound to
        port = s.getsockname()[1]
        return port


def is_port_available(port, host='localhost'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def resolve_submitit_port(requested_port, job_id):
    """Return one rendezvous port that is identical for every task in a job."""
    if requested_port:
        if not 1 <= requested_port <= 65535:
            raise ValueError(
                f"Distributed training port must be in [1, 65535], got "
                f"{requested_port}"
            )
        return requested_port

    # Submitit exposes the same job id to every task in a job. Deriving the
    # fallback from that id avoids choosing a different random port per node.
    job_id_text = str(job_id)
    digits = "".join(character for character in job_id_text if character.isdigit())
    if digits:
        port_seed = int(digits)
    else:
        port_seed = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(job_id_text)
        )
    return _SUBMITIT_PORT_BASE + port_seed % _SUBMITIT_PORT_COUNT


class SubmititLauncher:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        host_name = os.popen(
            "scontrol show hostnames $SLURM_JOB_NODELIST"
        ).read().split("\n")[0]
        job_env = self._set_gpu_args()
        port = resolve_submitit_port(self.args.port, job_env.job_id)
        if self.args.node_id == 0 and not is_port_available(port, host_name):
            raise RuntimeError(
                f"Distributed training port {port} is unavailable on master "
                f"host {host_name!r}. Pass a free port with --port."
            )
        # Using Accelerate for launching
        if self.args.num_nodes * self.args.gpu_per_node == 1:
            strategy = ""
        elif self.args.strategy == "ddp":
            strategy = "--multi_gpu"
        elif self.args.strategy == "ds2":
            strategy = "--use_deepspeed --deepspeed_config_file configs/ds_zero2.json"
        elif self.args.strategy == "ds3":
            strategy = "--use_deepspeed --deepspeed_config_file configs/ds_zero3.json"
        else:
            raise ValueError(f"Accelerate strategy only supports DDP and DeepSpeed")
        opts = " ".join(self.args.opts) if len(self.args.opts) > 0 else ""
        opts += f" num_gpu={self.args.num_nodes * self.args.gpu_per_node} "
        full_cfg_path = Path(self.args.config)
        cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
        cmd = f"OMP_NUM_THREADS={self.args.omp_num_threads} \
                TOKENIZERS_PARALLELISM={self.args.tokenizers_parallelism} \
                accelerate launch {strategy} \
                --mixed_precision {self.args.mixed_precision} \
                --num_machines {self.args.num_nodes} \
                --num_processes {self.args.gpu_per_node * self.args.num_nodes} \
                --num_cpu_threads_per_process {self.args.cpu_per_task} \
                --main_process_ip {host_name} \
                --main_process_port {port} \
                --machine_rank {self.args.node_id} \
                --dynamo_backend no \
                {self.args.run_file} \
                --config-path {cfg_path} \
                --config-name {cfg_file} \
                num_gpu={self.args.num_nodes * self.args.gpu_per_node} \
                hydra.run.dir=. \
                hydra.output_subdir=null \
                ~hydra.job_logging.handlers.file \
                hydra.job_logging.root.handlers=[console] \
                hydra/hydra_logging=disabled {opts}"
        subprocess.run(cmd, shell=True, check=True)

    def _set_gpu_args(self):
        job_env = submitit.JobEnvironment()
        self.args.job_dir = str(self.args.job_dir).replace("%j", job_env.job_id)
        # The executor launches one Submitit task per node. Its global task rank
        # is therefore the Accelerate machine rank.
        self.args.node_id = job_env.global_rank
        return job_env


def submitit_launch(args):
    """
    Multi node script launching with Submitit
    """
    additional_parameters = {}
    if args.nodelist != "":
        # if specifying node id
        nodelist = f"{str(args.nodelist)}"
        additional_parameters["nodelist"] = nodelist
    
    if args.exc_nodelist != "":
        # if specifying excluded node id
        exc_nodelist = f"{str(args.exc_nodelist)}"
        additional_parameters["exclude"] = exc_nodelist

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)
    executor.update_parameters(
        name=args.name,
        mem_gb=args.mem_per_gpu * args.gpu_per_node * args.num_nodes,
        gpus_per_node=args.gpu_per_node,
        tasks_per_node=1,
        cpus_per_task=args.gpu_per_node * args.cpu_per_task,
        nodes=args.num_nodes,
        slurm_qos=args.qos,
        slurm_partition=args.partition,
        slurm_account=args.account,
        slurm_time=args.time * 60,
        slurm_signal_delay_s=120,
        slurm_additional_parameters=additional_parameters
    )
    launcher = SubmititLauncher(args)
    job = executor.submit(launcher)
    print(f"Submitted job: {job.job_id}")


def accelerate_launch(args):
    """
    Single node script launching with Accelerate
    """
    opts = " ".join(args.opts) if len(args.opts) > 0 else ""
    opts += f" num_gpu={args.num_nodes * args.gpu_per_node} "
    if args.num_nodes * args.gpu_per_node == 1:
        strategy = ""
    elif args.strategy == "ddp":
        strategy = "--multi_gpu"
    elif args.strategy == "ds2":
        strategy = "--use_deepspeed --deepspeed_config_file configs/ds_zero2.json"
    elif args.strategy == "ds3":
        strategy = "--use_deepspeed --deepspeed_config_file configs/ds_zero3.json"
    else:
        raise ValueError(f"Accelerate strategy only supports DDP and DeepSpeed")
    full_cfg_path = Path(args.config)
    cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
    cmd = f"OMP_NUM_THREADS={args.omp_num_threads} \
            TOKENIZERS_PARALLELISM={args.tokenizers_parallelism} \
            accelerate launch {strategy} \
            --mixed_precision {args.mixed_precision} \
            --num_machines {args.num_nodes} \
            --num_processes {args.gpu_per_node * args.num_nodes} \
            --num_cpu_threads_per_process {args.cpu_per_task} \
            --main_process_port {args.port} \
            --dynamo_backend no \
            {args.run_file} \
            --config-path {cfg_path} \
            --config-name {cfg_file} \
            num_gpu={args.num_nodes * args.gpu_per_node} \
            hydra.run.dir=. \
            hydra.output_subdir=null \
            ~hydra.job_logging.handlers.file \
            hydra.job_logging.root.handlers=[console] \
            hydra/hydra_logging=disabled {opts}"
    subprocess.run(cmd, shell=True, check=True)


def python_launch(args):
    """
    Vanilla python launcher for degbugging purposes
    """
    opts = " ".join(args.opts) if len(args.opts) > 0 else ""
    full_cfg_path = Path(args.config)
    cfg_path, cfg_file = str(full_cfg_path.parent), str(full_cfg_path.name)
    cmd = f"OMP_NUM_THREADS={args.omp_num_threads} \
            TOKENIZERS_PARALLELISM={args.tokenizers_parallelism} \
            python {args.run_file} \
            --config-path {cfg_path} \
            --config-name {cfg_file} \
            num_gpu=1 \
            hydra.run.dir=. \
            hydra.output_subdir=null \
            ~hydra.job_logging.handlers.file \
            hydra.job_logging.root.handlers=[console] \
            hydra/hydra_logging=disabled {opts}"
    subprocess.run(cmd, shell=True, check=True)
