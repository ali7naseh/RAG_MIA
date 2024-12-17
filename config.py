from dataclasses import dataclass
from typing import Optional
from simple_parsing.helpers import Serializable


@dataclass
class RAGConfig(Serializable):
    """
        Configuration for the RAG system
    """
    retriever: str
    """Retriever model to use"""
    eval_dataset: str
    """BEIR dataset to evaluate"""
    split: str = "test"
    """Split of the dataset to use"""
    retrieve_k: int = 3
    """Number of docs for each query in RAG"""
    reranker: Optional[str] = None
    """Re-ranker model to use, if any"""
    rerank_k: Optional[int] = None
    """Top-k documents selected after re-ranking"""
    def __post_init__(self):
        if self.reranker is not None:
            if self.rerank_k is None:
                raise ValueError("Please provide a value for rerank_k")
            if self.rerank_k >= self.retrieve_k:
                raise ValueError("rerank_k should be less than than retrieve_k")


@dataclass
class AttackConfig(Serializable):
    """
        Configuration for the attack
    """
    attack_method: str
    """Attack method to use"""
    repeat_times: int = 10
    """Repeat several times to compute average"""
    M: int = 10
    """One of our parameters, the number of target docs"""
    N: int = 2
    """Number of target queries generated for each doc"""
    top_k: int = 5
    """Number of questions after filtering"""
    seed: int = 12
    """Random seed"""
    name: str = "debug"
    """Name of log and result"""
    from_ckpt: bool = False
    """Load from checkpoint if this flag is set"""
    post_filter: Optional[str] = None
    """Do post filtering"""
    evaluate_attack: Optional[bool] = True
    """If False, attack only generates relevant data for the attack but does not actually evaluate it"""


@dataclass
class LLMConfig(Serializable):
    """
        Configuration for the LLM system
    """
    model_name: str
    """Name of the model"""
    model_config_path: Optional[str] = None
    """Path to the model config"""
    shadow_model_name: str = "llama3"
    """Name of the shadow model"""
    gpu_id: int = 0
    """GPU ID to use"""


@dataclass
class ExperimentConfig(Serializable):
    """
        Configuration for the experiment
    """
    rag_config: RAGConfig
    """Configuration for the RAG system"""
    attack_config: AttackConfig
    """Configuration for the attack"""
    llm_config: LLMConfig
    """Configuration for the LLM system"""

    def get_log_name(self):
        log_name = f"{self.attack_config.attack_method}-{self.rag_config.eval_dataset}-{self.llm_config.model_name}-{self.llm_config.shadow_model_name}-{self.rag_config.retriever}-R{self.rag_config.retrieve_k}-Top{self.attack_config.top_k}-M{self.attack_config.M}-N{self.attack_config.N}"
        return log_name
