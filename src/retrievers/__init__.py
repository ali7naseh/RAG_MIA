from .Contriever import Contriever
from .ColBERT import ColBERT
from .GTE import GTE
from .BGE import BGE
from .NVEmbed import NVEmbed
from .Jina import JinaReranker
from .MixedBread import MixedBreadReranker


def create_retriever(name: str, dataset: str):
    """
    Method to create a retriever object
    """
    provider = name.lower()

    if provider == 'contriever':
        model = Contriever
    elif provider == 'colbert':
        model = ColBERT
    elif provider == 'gte':
        model = GTE
    elif provider == 'bge':
        model = BGE
    elif provider == 'nvembed':
        model = NVEmbed
    elif provider == 'jina':
        model = JinaReranker
    elif provider == 'mixedbread':
        model = MixedBreadReranker
    else:
        raise ValueError(f"ERROR: Unknown retriever {name}")
    return model(dataset)
