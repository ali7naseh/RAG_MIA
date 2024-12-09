import argparse
import os
import json
import random
from src.models import create_model
from src.utils import setup_seeds
from beir.datasets.data_loader import GenericDataLoader
# from beir.retrieval import models
from mia_utils.mia import MIA_Attacker
from mia_utils.direct_query import Direct_Query_Attacker
from mia_utils.s2 import S2_Attacker
from mia_utils.mba import MBA_Attacker
from mia_utils.utils import get_superset_file


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
    parser.add_argument("--from_ckpt", action="store_true", help="Load from checkpoint if this flag is set.")
    parser.add_argument("--post_filter", type=str, help="Do post filtering")
    parser.add_argument('--retrieve_k', type=int, default=3, help='num of docs for each query in rag')
    

    args = parser.parse_args()
    print(args)
    return args

def main(args):
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
        args.shadow_model_config_path = f'model_configs/{args.shadow_model_name}_config.json'

    # load target queries and answers
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)
    try:
        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
    except Exception as e:
        corpus_path = os.path.join(data_path, 'corpus.json')
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)

    # Load the selected indices from selected_indices.json
    selected_indices_file = f"./datasets/{args.eval_dataset}/selected_indices.json"
    with open(selected_indices_file, 'r') as f:
        selected_indices = json.load(f)

    if args.post_filter is not None and args.post_filter != 'None' and args.attack_method in ['mia']:
        pass
    elif not args.from_ckpt:
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
    else:
        # Load the existing file
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{args.name}.json'
        with open(output_file, 'r') as f:
            target_docs = json.load(f)
        # Remove non-member documents from the corpus
        for doc_id, doc_content in target_docs.items():
            if doc_content['mem'] == 'no' and doc_id in corpus:
                corpus.pop(doc_id)

    print(args.shadow_model_config_path, args.model_config_path)
    
    if args.attack_method in ['direct_query']:

        attacker = Direct_Query_Attacker(
            args,
            target_docs=target_docs,
            corpus=corpus
        )

        #target LLM
        llm = create_model(args.model_config_path)
        llm.to(llm.device)

        attacker.generate_questions()
        attacker.retrieve_docs_()
        attacker.query_target_llm(llm=llm, from_ckpt=args.from_ckpt)
        attacker.calculate_score()

    elif args.attack_method in ['s2']:
        attacker = S2_Attacker(
            args,
            target_docs=target_docs,
            corpus=corpus
        )

        #target LLM
        llm = create_model(args.model_config_path)
        llm.to(llm.device)

        attacker.generate_questions()
        attacker.retrieve_docs_()
        attacker.query_target_llm(llm=llm, from_ckpt=args.from_ckpt)
        attacker.calculate_score()

    elif args.attack_method in ['mia']:

        if args.post_filter is not None and args.post_filter != 'None':

            if args.post_filter == 'top':
                superset_filename = get_superset_file(args, random=False)
            elif args.post_filter == 'random':
                args.name = 'random-'+args.name
                superset_filename = get_superset_file(args, random=True)
                
            with open(superset_filename, 'r') as file:
                data = json.load(file)

            # Initialize the MIA_Attacker
            attacker = MIA_Attacker(
                args=args,
                model=None,
                target_docs=data,
                corpus=None
            )
            # Perform post-filtering
            attacker.post_filter(top_k=args.top_k, mode=args.post_filter)
            attacker.calculate_score()

        else:
   
            attacker = MIA_Attacker(
                args,
                target_docs=target_docs,
                corpus=corpus
            )

            if not args.from_ckpt:
                #questions from GPT-4
                query_file = f"datasets/{args.eval_dataset}/clean_data_with_questions.json"
                with open(query_file, 'r') as f:
                    source_docs = json.load(f)
                attacker.generate_questions_GPT_4(source_docs)

                #Use IR to filter question- pre-filtering
                attacker.filter_questions_topk(top_k=args.top_k)

            attacker.retrieve_docs_(k=args.retrieve_k, retriever = args.retriever)
            #generate groud truth answers
            validate_llm = create_model(args.shadow_model_config_path)
            validate_llm.to(validate_llm.device)
            attacker.generate_ground_truth_answers(validate_llm, from_ckpt=args.from_ckpt)
            del validate_llm
        
            #target LLM
            llm = create_model(args.model_config_path)
            llm.to(llm.device)

            attacker.query_target_llm(llm=llm, from_ckpt=args.from_ckpt)
            attacker.calculate_score()

    elif args.attack_method in ['mba']:
        # Authors tested num_masks in [5, 10, 15, 20] and picked M with the higest AUC
        attacker = MBA_Attacker(
            args,
            target_docs=target_docs,
            corpus=corpus,
            num_masks=10
        )

        #target LLM
        llm = create_model(args.model_config_path)
        llm.to(llm.device)

        attacker.generate_questions()
        attacker.retrieve_docs_()
        attacker.query_target_llm(llm=llm, from_ckpt=args.from_ckpt)
        attacker.calculate_score()

    else:
        raise ValueError(f"Invalid attack method: {args.attack_method}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
