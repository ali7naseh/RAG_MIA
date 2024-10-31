import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)
    cmd = f"nohup python3 -u main_mia.py \
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --model_name {test_params['model_name']}\
        --shadow_model_name {test_params['shadow_model_name']}\
        --top_k {test_params['top_k']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --N {test_params['N']}\
        --seed {test_params['seed']}\
        --name {log_name} " + \
        ("--from_ckpt " if test_params['from_ckpt'] else "") + \
        f"> {log_file}"
        
    os.system(cmd)

def get_log_name(test_params):
    os.makedirs(f"logs", exist_ok=True)
    # Generate a log file name
    log_name = f"{test_params['eval_dataset']}-{test_params['model_name']}-Top{test_params['top_k']}-M{test_params['M']}-N{test_params['N']}"
    return f"logs/{log_name}.txt", log_name

test_params = {
    # beir_info
    'retriever': "colbert", #retriever
    'eval_dataset': "nfcorpus",
    'split': "test",

    # LLM setting
    'model_name': 'llama3', #target model
    'shadow_model_name': 'llama3', # shadow model
    'N': 10, # how many questions generated for each target doc 
    'gpu_id': 0,

    # attack
    'attack_method': 'aa',
    'repeat_times': 1,
    'M': 1000, #how many target docs
    'top_k': 10, # how many questions each doc after filtering
    'seed': 12,
    'from_ckpt': True
}

# for dataset in ['nq', 'hotpotqa', 'msmarco']:
for dataset in [ 'nfcorpus']:
    test_params['eval_dataset'] = dataset
    run(test_params)