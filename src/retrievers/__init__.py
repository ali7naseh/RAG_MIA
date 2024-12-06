from .Contriever import Contriever
from .ColBERT import ColBERT


def create_retriever(provider: str, dataset: str):
    """
    Method to createe a retriever object
    """
    provider_ = provider.lower()

    if provider_ == 'contriever':
        model = Contriever
    elif provider_ == 'colbert':
        model = ColBERT
    else:
        raise ValueError(f"ERROR: Unknown retriever {provider}")
    return model(dataset)
