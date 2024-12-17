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
from pathlib import Path
from config import ExperimentConfig

from simple_parsing import ArgumentParser


def parse_config():
    parser = argparse.ArgumentParser(description='test')

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Path to attack config file", type=Path)
    args, remaining_argv = parser.parse_known_args()
    # Attempt to extract as much information from config file as you can
    config = ExperimentConfig.load(args.config, drop_extra_fields=False)
    # Also give user the option to provide config values over CLI
    parser = ArgumentParser(parents=[parser])
    parser.add_arguments(ExperimentConfig, dest="exp_config", default=config)
    args = parser.parse_args(remaining_argv)
    config: ExperimentConfig = args.exp_config

    # Compute what the name should be
    config.attack_config.name = config.get_log_name()

    # Make sure log directory exists
    os.makedirs("logs", exist_ok=True)

    print(json.dumps(json.loads(config.dumps_json()), indent=4))
    return config


def main(config: ExperimentConfig):
    device = 'cuda'
    setup_seeds(config.attack_config.seed)
    if config.llm_config.model_config_path == None:
        config.llm_config.model_config_path = f'model_configs/{config.llm_config.model_name}_config.json'
        shadow_model_config_path = f'model_configs/{config.llm_config.shadow_model_name}_config.json'

    # load target queries and answers
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, config.rag_config.eval_dataset)
    try:
        corpus, queries, _ = GenericDataLoader(data_path).load(split="test")
    except Exception as e:
        corpus_path = os.path.join(data_path, 'corpus.json')
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)

    # Load the selected indices from selected_indices.json
    selected_indices_file = f"./datasets/{config.rag_config.eval_dataset}/selected_indices.json"
    with open(selected_indices_file, 'r') as f:
        selected_indices = json.load(f)

    if config.attack_config.post_filter is not None and config.attack_config.post_filter != 'None' and config.attack_config.attack_method in ['mia']:
        pass
    elif not config.attack_config.from_ckpt:
        # Extract mem and nonmem indices
        mem_indices = selected_indices.get('mem_indices', [])
        nonmem_indices = selected_indices.get('non_mem_indices', [])
        # Ensure that mem_indices and nonmem_indices are valid and contain enough elements
        assert len(mem_indices) >= config.attack_config.M * config.attack_config.repeat_times, "Not enough mem indices to sample from"
        assert len(nonmem_indices) >= config.attack_config.M * config.attack_config.repeat_times, "Not enough non-mem indices to sample from"
        # Sample from nonmem_indices
        docs_nonmem_id = random.sample(nonmem_indices, config.attack_config.M * config.attack_config.repeat_times)
        nonmem_docs = {doc_id: {**corpus.pop(doc_id), 'mem': 'no'} for doc_id in docs_nonmem_id}
        # Sample from mem_indices
        sampled_docs_mem_id = random.sample(mem_indices, config.attack_config.M * config.attack_config.repeat_times)
        mem_docs = {doc_id: {**corpus[doc_id], 'mem': 'yes'} for doc_id in sampled_docs_mem_id}
        # Combine the mem and nonmem documents into the target_docs dictionary
        target_docs = {**mem_docs, **nonmem_docs}
    else:
        # Load the existing file
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{config.attack_config.name}.json'
        with open(output_file, 'r') as f:
            target_docs = json.load(f)
        # Remove non-member documents from the corpus
        for doc_id, doc_content in target_docs.items():
            if doc_content['mem'] == 'no' and doc_id in corpus:
                corpus.pop(doc_id)

    print(shadow_model_config_path, config.llm_config.model_config_path)
    
    if config.attack_config.attack_method in ['direct_query']:

        attacker = Direct_Query_Attacker(
            config,
            target_docs=target_docs,
            corpus=corpus
        )

        attacker.generate_questions()
        attacker.retrieve_docs_()
        if config.attack_config.evaluate_attack:
            #target LLM
            llm = create_model(config.llm_config.model_config_path)
            llm.to(llm.device)

            attacker.query_target_llm(llm=llm, from_ckpt=config.attack_config.from_ckpt)
            attacker.calculate_score()

    elif config.attack_config.attack_method in ['s2']:
        attacker = S2_Attacker(
            config,
            target_docs=target_docs,
            corpus=corpus
        )

        attacker.generate_questions()
        attacker.retrieve_docs_()
        if config.attack_config.evaluate_attack:
            #target LLM
            llm = create_model(config.llm_config.model_config_path)
            llm.to(llm.device)

            attacker.query_target_llm(llm=llm, from_ckpt=config.attack_config.from_ckpt)
            attacker.calculate_score()

    elif config.attack_config.attack_method in ['mia']:

        if config.attack_config.post_filter is not None and config.attack_config.post_filter != 'None':

            if config.attack_config.post_filter == 'top':
                superset_filename = get_superset_file(config, random=False)
            elif config.attack_config.post_filter == 'random':
                config.attack_config.name = f'random-{config.attack_config.name}'
                superset_filename = get_superset_file(config, random=True)
                
            with open(superset_filename, 'r') as file:
                data = json.load(file)

            # Initialize the MIA_Attacker
            attacker = MIA_Attacker(
                config=config,
                model=None,
                target_docs=data,
                corpus=None
            )
            # Perform post-filtering
            attacker.post_filter(top_k=config.attack_config.top_k, mode=config.attack_config.post_filter)
            if config.attack_config.evaluate_attack:
                attacker.calculate_score()

        else:
   
            attacker = MIA_Attacker(
                config,
                target_docs=target_docs,
                corpus=corpus
            )

            if not config.attack_config.from_ckpt:
                #questions from GPT-4
                query_file = f"datasets/{config.rag_config.eval_dataset}/clean_data_with_questions.json"
                with open(query_file, 'r') as f:
                    source_docs = json.load(f)
                attacker.generate_questions_GPT_4(source_docs)

                #Use IR to filter question- pre-filtering
                attacker.filter_questions_topk(top_k=config.attack_config.top_k)

            attacker.retrieve_docs_()
            #generate ground truth answers
            validate_llm = create_model(shadow_model_config_path)
            validate_llm.to(validate_llm.device)
            attacker.generate_ground_truth_answers(validate_llm, from_ckpt=config.attack_config.from_ckpt)
            del validate_llm
        
            if config.attack_config.evaluate_attack:
                #target LLM
                llm = create_model(config.llm_config.model_config_path)
                llm.to(llm.device)

                attacker.query_target_llm(llm=llm, from_ckpt=config.attack_config.from_ckpt)
                attacker.calculate_score()

    elif config.attack_config.attack_method in ['mba']:
        # Authors tested num_masks in [5, 10, 15, 20] and picked M with the higest AUC
        attacker = MBA_Attacker(
            config,
            target_docs=target_docs,
            corpus=corpus,
            num_masks=10
        )

        if not config.attack_config.from_ckpt:
            attacker.generate_questions()
            attacker.retrieve_docs_()
        
        if config.attack_config.evaluate_attack:
            #target LLM
            llm = create_model(config.llm_config.model_config_path)
            llm.to(llm.device)

            attacker.query_target_llm(llm=llm, from_ckpt=config.attack_config.from_ckpt)
            attacker.calculate_score()

    else:
        raise ValueError(f"Invalid attack method: {config.attack_config.attack_method}")


if __name__ == '__main__':
    config = parse_config()
    main(config)
