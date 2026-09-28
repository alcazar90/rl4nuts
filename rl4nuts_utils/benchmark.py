import shlex
import subprocess

from typing import List
from dataclasses import dataclass
import tyro


@dataclass
class Args:
    env_ids: List[str]
    """the ids of the environment to compare"""
    command: str
    """the command to run"""
    num_seeds: int = 3
    """the number of random seeds"""
    start_seed: int = 1
    """the number of the starting seed"""
    workers: int = 0
    """the number of workers to run benchmark experiments in parallel"""
    auto_tag: bool = True
    """if toggled, the runs will be tagged with git tags, commit, and pull request number if possible"""

def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"Running command: {command}")

    # Use subprocess.PIPE to capture the output
    fd = subprocess.Popen(
        command_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, errors = fd.communicate()

    return_code = fd.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"

    # Convert bytes to string and strip leading/trailing whitespaces
    return output.decode("utf-8").strip()



if __name__ == "__main__":
    args = tyro.cli(Args)

    # Cartesian product to generate all experiment commands (seed X env_id)
    commands = []
    for seed in range(0, args.num_seeds):
        for env_id in args.env_ids:
            commands += [
                " ".join(
                    [
                        args.command,
                        "--env-id",
                        env_id,
                        "--seed",
                        str(seed + args.start_seed),
                    ]
                )
            ]

    print(f"======= commands to run ({args.num_seeds} seeds X {len(args.env_ids)} envs = {args.num_seeds * len(args.env_ids)}):")
    for command in commands:
        print(command)

    if args.workers > 0:
        print(f"Running the experiments with {args.workers} workers...")
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="rl4nut-benchmark-worker-",
            )

        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)

    else:
        print("(dry run) not running the experiments because --workers is set to 0; just printing the commands to run")
