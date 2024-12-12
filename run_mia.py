import os
import argparse
from config import ExperimentConfig


def run(config_path):
    log_file, log_name = get_log_name(config_path)
    cmd = f"nohup python3 -u main_mia.py --config {config_path} --name {log_name} > {log_file}"
    os.system(cmd)


def get_log_name(config_path):
    config = ExperimentConfig.load(config_path)
    log_name = config.get_log_name()
    os.makedirs("logs", exist_ok=True)
    return f"logs/{log_name}.txt", log_name


def main():
    parser = argparse.ArgumentParser(description="Run the main script with a specified config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    run(args.config)

if __name__ == "__main__":
    main()
