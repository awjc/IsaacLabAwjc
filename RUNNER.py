import subprocess
import sys
import os
import re

TASK_MAP = {
    "cartpole": "Isaac-Cartpole-v0",
    "ant": "Isaac-Ant-v0",
    "humanoid": "Isaac-Humanoid-v0"
}


TASK = "humanoid"


NUM_ENVS_MAP = {
    "cartpole": 1024,
    "ant": 512,
    "humanoid": 128
}


# ACTION = "play"


ACTION = sys.argv[1] if len(sys.argv) > 1 else "train"
NUM_ENVS = sys.argv[2] if len(sys.argv) > 2 else (1 if "play" in ACTION else NUM_ENVS_MAP[TASK])


RUN_FOLDER = f"logs/rsl_rl/{TASK}/"
CHECKPOINT_SUFFIX = ".pt"

RUN = None
CHECKPOINT = None



# RUN = "2024-12-06_18-17-56"
# CHECKPOINT = "model_2100.pt"


def get_last_run(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    last_subfolder = sorted(subfolders)[-1]
    return last_subfolder


def get_last_checkpoint(run_id):
    folder_path = RUN_FOLDER + run_id
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = [f for f in files if f.endswith(CHECKPOINT_SUFFIX)]

    def numeric_sort(file):
        match = re.search(r'(\d+)', file)
        return int(match.group(0)) if match else float('inf')

    sorted_files = sorted(files, key=numeric_sort)
    return sorted_files[-1]


def run_isaaclab():
    cmd = \
        "cmd.exe /c isaaclab.bat -p " \

    if 'train' in ACTION:
        cmd += "source/standalone/workflows/rsl_rl/train.py "
    if 'play' in ACTION:
        cmd += "source/standalone/workflows/rsl_rl/play.py "

    cmd += \
        f"--task {TASK_MAP[TASK]} " \
        f"--num_envs {NUM_ENVS} "

    if ACTION in ('train-cont', 'play'):
        if ACTION == 'train-cont':
            cmd += "--resume RESUME "

        if RUN and CHECKPOINT:
            cmd += \
                f'--load_run {RUN} ' \
                f'--checkpoint {CHECKPOINT} '
        else:
            last_run = get_last_run(RUN_FOLDER)
            last_checkpoint = get_last_checkpoint(last_run)
            cmd += \
                f"--load_run {last_run} " \
                f'--checkpoint {last_checkpoint} '

    # Execute the command
    print(">>> COPYING CMD:")
    print('~~~~' * 5)
    print(f"{cmd}")
    print('~~~~' * 5)

    subprocess.run(f'echo {cmd} | clip.exe', shell=True)





















    # with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    #     try:
    #         # Stream stdout
    #         if process.stdout is not None:
    #             for line in iter(process.stdout.readline, ''):
    #                 print(line, end='')  # Print each line as it comes in

    #         # Optionally, you can also stream stderr
    #         if process.stderr is not None:
    #             for line in iter(process.stderr.readline, ''):
    #                 print(line, end='', file=sys.stderr)

    #     except KeyboardInterrupt:
    #         # Handle Ctrl+C and terminate the process
    #         print("\nProcess interrupted, terminating...")
    #         process.terminate()
    #         process.wait()  # Ensure the process terminates gracefully

    #     finally:
    #         # Cleanup in case the process finishes normally
    #         if process.poll() is None:
    #             process.terminate()
    #             process.wait()

    # result = subprocess.run(command, shell=True, text=True, capture_output=True)

    # print("STDOUT:")
    # print(result.stdout)

    # print("STDERR:")
    # print(result.stderr)

if __name__ == '__main__':
    run_isaaclab()
