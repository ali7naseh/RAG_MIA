import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score
from src.attack import Attacker
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from mia_utils.mia import MIA_Attacker
from src.prompts import wrap_prompt
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument('--retriever', type=str, default="colbert", help='') #reserved
    parser.add_argument('--eval_dataset', type=str, default="nfcorpus", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--shadow_model_name', type=str, default='llama3')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='') #reserved
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target docs')
    parser.add_argument('--N', type=int, default=2, help='number of target queries generated for each doc')
    parser.add_argument('--top_k', type=int, default=5,help='num of questions after filtering' )
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
        args.shadow_model_config_path = f'model_configs/{args.shadow_model_name}_config.json'

    # load target queries and answers
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)
    corpus, queries, _ = GenericDataLoader(data_path).load(split="test")

    # Load the selected indices from selected_indices.json
    selected_indices_file = f"./datasets/{args.eval_dataset}/selected_indices.json"
    with open(selected_indices_file, 'r') as f:
        selected_indices = json.load(f)

    # Extract mem and nonmem indices
    mem_indices = selected_indices.get('mem_indices', [])
    nonmem_indices = selected_indices.get('non_mem_indices', [])
    # Ensure that mem_indices and nonmem_indices are valid and contain enough elements
    assert len(mem_indices) >= args.M * args.repeat_times, "Not enough mem indices to sample from"
    assert len(nonmem_indices) >= args.M * args.repeat_times, "Not enough non-mem indices to sample from"
    # Sample from nonmem_indices
    docs_nonmem_id = random.sample(nonmem_indices, args.M * args.repeat_times)
    nonmem_docs = {doc_id: {**corpus.pop(doc_id), 'mem': 'no'} for doc_id in docs_nonmem_id}
    # Sample from mem_indices
    sampled_docs_mem_id = random.sample(mem_indices, args.M * args.repeat_times)
    mem_docs = {doc_id: {**corpus[doc_id], 'mem': 'yes'} for doc_id in sampled_docs_mem_id}
    # Combine the mem and nonmem documents into the target_docs dictionary
    target_docs = {**mem_docs, **nonmem_docs}


    print(args.shadow_model_config_path,args.model_config_path)
    shadow_llm = create_model(args.shadow_model_config_path)
    shadow_llm.model.to(shadow_llm.device)
    
    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        if args.retriever == 'colbert':
            retriever = None # retrival is done in another repo
        else:
            #TBD
            pass

        attacker = MIA_Attacker(args,
                            model=retriever,
                            shadow_llm=shadow_llm,
                            target_docs=target_docs,
                            corpus=corpus
                            ) 

        #questions from GPT-4
        query_file = f"./clean_data_with_questions.json"
        with open(query_file, 'r') as f:
            source_docs = json.load(f)
        attacker.generate_questions_GPT_4(source_docs)
        del attacker.shadow_llm
        del shadow_llm

        #Use IR to filter question
        attacker.filter_questions_topk(top_k=args.top_k)

        #generate groud truth answers
        validate_llm = create_model(args.model_config_path)
        validate_llm.model.to(validate_llm.device)
        attacker.generate_ground_truth_answers(validate_llm)
        attacker.retrieve_docs_()
    
    #target LLM
    llm = create_model(args.model_config_path)
    llm.model.to(llm.device)

    # Load the existing file
    output_dir = 'results/target_docs'
    output_file = f'{output_dir}/{args.name}.json'
    with open(output_file, 'r') as f:
        data = json.load(f)

    # Query the target LLM
    for doc_id, doc_content in data.items():
        num_questions = len(doc_content.get('questions', []))
        for i, question in enumerate(doc_content['questions']):
            # Get the top-k doc IDs (in this case, it's just one top doc)
            retrieved_doc_ids = doc_content['retrieved_doc_ids']
            topk_contents = [corpus[doc]['text'] for doc in retrieved_doc_ids[i]]  # Iterate over the document IDs directly
            query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)
            response = llm.query(query_prompt)

            if 'llm_responses' not in doc_content:
                doc_content['llm_responses'] = []
            doc_content['llm_responses'].append(response)

    # Save the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    attacker.calculate_accuracy_()

if __name__ == '__main__':
    main()