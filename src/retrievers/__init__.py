from .Contriever import Contriever
from .ColBERT import ColBERT
from .GTE import GTE
from .BGE import BGE
from .NVEmbed import NVEmbed


def create_retriever(name: str, dataset: str):
    """
    Method to createe a retriever object
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
    else:
        raise ValueError(f"ERROR: Unknown retriever {name}")
    return model(dataset)
