import os
import yaml
import argparse

def run(test_params):
    log_file, log_name = get_log_name(test_params)
    cmd = f"nohup python3 -u main_mia.py \
        --eval_dataset {test_params['eval_dataset']} \
        --split {test_params['split']} \
        --model_name {test_params['model_name']} \
        --retriever {test_params['retriever']} \
        --shadow_model_name {test_params['shadow_model_name']} \
        --top_k {test_params['top_k']} \
        --gpu_id {test_params['gpu_id']} \
        --attack_method {test_params['attack_method']} \
        --repeat_times {test_params['repeat_times']} \
        --M {test_params['M']} \
        --N {test_params['N']} \
        --seed {test_params['seed']} \
        --retrieve_k {test_params['retrieve_k']} \
        --name {log_name} \
        --post_filter {test_params['post_filter']} "  + \
        ("--from_ckpt " if test_params['from_ckpt'] else "") + \
        f"> {log_file}"
        
    os.system(cmd)

def get_log_name(test_params):
    os.makedirs("logs", exist_ok=True)
    log_name = f"{test_params['attack_method']}-{test_params['eval_dataset']}-{test_params['model_name']}-{test_params['shadow_model_name']}-{test_params['retriever']}-R{test_params['retrieve_k']}-Top{test_params['top_k']}-M{test_params['M']}-N{test_params['N']}"
    return f"logs/{log_name}.txt", log_name

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['test_params']

def main():
    parser = argparse.ArgumentParser(description="Run the main script with a specified config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    test_params = load_config(args.config)

    run(test_params)

if __name__ == "__main__":
    main()
