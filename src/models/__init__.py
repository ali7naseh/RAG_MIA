# from .PaLM2 import PaLM2
# from .Vicuna import Vicuna
from .GPT import GPT
from .Llama import Llama
from .Llama3 import Llama3
from .Llama3_70b import Llama3_70b
from .Gemma2 import Gemma2_2B, Gemma2_9B
import json


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'palm2':
        model = PaLM2(config)
    elif provider == 'vicuna':
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'llama3':
        model = Llama3(config)
    elif provider == 'llama3_70b':
        model = Llama3_70b(config)
    elif provider == 'gemma2_2b':
        model = Gemma2_2B(config)
    elif provider == 'gemma2_9b':
        model = Gemma2_9B(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model
