import subprocess
import sys
import time
import os

EXPERIMENTS = [
    ["--algo", "ppo", "--role", "pursuer", "--n_envs", "4"],
    ["--algo", "sac", "--role", "pursuer", "--her", "--n_envs", "4"],
    ["--algo", "td3", "--role", "pursuer", "--her", "--n_envs", "4"],
    ["--algo", "ppo", "--role", "evader", "--n_envs", "4"],
    ["--algo", "sac", "--role", "evader", "--n_envs", "4"],
    ["--algo", "td3", "--role", "evader", "--n_envs", "4"],
]

def main():
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "train_curriculum.py")
    print("Launching 6 Parallel Training Windows...")
    for exp in EXPERIMENTS:
        cmd = f'start "Train {exp[1]} {exp[3]}" cmd /k "{python_exe} {script_path} {" ".join(exp)}"'
        subprocess.Popen(cmd, shell=True)
        time.sleep(2)
    print("Launching TensorBoard...")
    time.sleep(3)
    tb_cmd = f'start "TensorBoard" cmd /k "{python_exe} -m tensorboard.main --logdir ./logs"'
    subprocess.Popen(tb_cmd, shell=True)
    print("\nAll processes launched!")
    print("Check your taskbar for 6 new CMD windows + 1 TensorBoard window.")
    print("You can monitor training live at http://localhost:6006")

if __name__ == "__main__":
    main()
