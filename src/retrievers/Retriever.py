
class Retriever:
    def __init__(self, name, eval_dataset: str):
        self.name = name
        self.eval_dataset = eval_dataset
        self.model = None

    def search_question(self, question: str, k: int):
        raise NotImplementedError("search_question method must be implemented in subclass")
