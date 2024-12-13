"""
    Load up a given yaml config file and convert it into an ExperimentConfig.
"""
from config import ExperimentConfig, AttackConfig, LLMConfig, RAGConfig
import yaml
import sys


def main(filepath: str):
    # Load up a generic yaml file at the provided path
    # Note that this has all entries flat. We need to map them properly into the desired format
    with open(filepath, 'r') as file:
        raw_config = yaml.safe_load(file)
    raw_config = raw_config["test_params"]
    
    rag_config = RAGConfig(
        retriever=raw_config.get("retriever"),
        eval_dataset=raw_config.get("eval_dataset"),
        split=raw_config.get("split", "test"),
        retrieve_k=raw_config.get("retrieve_k", 3)
    )

    attack_config = AttackConfig(
        attack_method=raw_config.get("attack_method", ""),
        repeat_times=raw_config.get("repeat_times", 10),
        M=raw_config.get("M", 10),
        N=raw_config.get("N", 2),
        top_k=raw_config.get("top_k", 5),
        seed=raw_config.get("seed", 12),
        from_ckpt=raw_config.get("from_ckpt", False),
        post_filter=raw_config.get("post_filter", None)
    )

    llm_config = LLMConfig(
        model_name=raw_config.get("model_name", "palm2"),
        shadow_model_name=raw_config.get("shadow_model_name", "llama3"),
        gpu_id=raw_config.get("gpu_id", 0)
    )

    experiment_config = ExperimentConfig(
        rag_config=rag_config,
        attack_config=attack_config,
        llm_config=llm_config
    )

    # Save this config to a new yaml file
    with open(filepath, 'w') as file:
        yaml.dump(experiment_config.to_dict(), file)


if __name__ == "__main__":
    config = sys.argv[1] # Example trec-covid_mba.yml
    main(config)
