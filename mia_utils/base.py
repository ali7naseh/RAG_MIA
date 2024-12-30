import os
import json
from tqdm import tqdm
from src.retrievers import create_retriever
from src.prompts import wrap_prompt


class BaseAttacker:
    """
        Base class for all attackers.
    """
    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.attack_method = self.config.attack_config.attack_method

        # reserved for retriever
        self.model = kwargs.get('model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)

        self.shadow_llm = kwargs.get('shadow_llm', None)
        self.target_docs = kwargs.get('target_docs', None)

        #load dataset
        self.corpus = kwargs.get('corpus', None)
        self.document_slice_size = 2048

    def prompt_len(self, tokenizer, question: str, doc_id: int, question_id: int):
        # Default to None: use model-specific max_tokens
        return None

    def save_target_docs(self):
        # Define the output directory and filename dynamically
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.config.attack_config.name}.json'
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save the target_docs dictionary to the specified JSON file
        with open(output_file, 'w') as f:
            json.dump(self.target_docs, f, indent=4)

    def retrieve_docs_(self, k: int = 5, retriever: str = 'colbert', from_ckpt: bool = True):
        # Set output file for checkpointing
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.config.attack_config.name}.json'
        with open(output_file, 'r') as f:
            self.target_docs = json.load(f)

        # Create the retriever
        retriever = create_retriever(retriever, self.config.rag_config.eval_dataset)

        for doc_id, doc_content in tqdm(self.target_docs.items(), desc="Processing documents"):
            questions = doc_content.get('questions', [])
            retrieved_doc_ids = doc_content.get('retrieved_doc_ids', [])
            if not from_ckpt:
                retrieved_doc_ids = []  # Start fresh if `from_ckpt` is False

            if len(retrieved_doc_ids) < len(questions):
                print(f"Reprocessing document: {doc_id}")
                doc_content['retrieved_doc_ids'] = []  # Clear existing results to start fresh
                retrieved_doc_ids = doc_content['retrieved_doc_ids']

                # Retrieve documents for all questions
                for i, question in enumerate(questions):
                
                    doc_ids = retriever.search_question(question, k)
                    if doc_ids is None:
                        print(f"Error during retrieval for question '{question}' in doc {doc_id}")
                        continue
                    else:
                        print(f"Retrieved document IDs for question '{question}': {doc_ids}")
                    # Append retrieved document IDs
                    retrieved_doc_ids.append(doc_ids)

                # Update the target document
                doc_content['retrieved_doc_ids'] = retrieved_doc_ids

                # Save progress after processing the target document
                self.save_target_docs()
                print(f"Checkpoint saved for doc_id: {doc_id}")
            else:
                print(f"All questions already processed for doc {doc_id}")

        # Final save after processing all documents
        self.save_target_docs()
        print("All documents processed and saved.")

    # def retrieve_docs_(self):
    #     k = self.config.rag_config.retrieve_k
    #     retriever = self.config.rag_config.retriever
    #     # Create retriever
    #     retriever = create_retriever(retriever, self.config.rag_config.eval_dataset)

    #     # Create reranker, if specified
    #     reranker_model = None
    #     rerank_k = self.config.rag_config.rerank_k
    #     if self.config.rag_config.reranker is not None:
    #         reranker_model = create_retriever(self.config.rag_config.reranker, self.config.rag_config.eval_dataset)

    #     for doc_id, doc_content in self.target_docs.items():
    #         questions = doc_content.get('questions', [])
    #         retrieved_doc_ids = []

    #         print(f"Retrieving documents for doc_id: {doc_id}")

    #         for question in questions:
    #             # Perform the search for each question
    #             print(f"Querying for question: {question}")

    #             doc_ids = retriever.search_question(question, k)
    #             if doc_ids is None:
    #                 print(f"Error during retrieval for question '{question}' in doc {doc_id}")
    #                 continue
    #             else:
    #                 print(f"Retrieved document IDs for question '{question}': {doc_ids}")
                
    #             # Rerank the retrieved documents
    #             if reranker_model is not None:
    #                 retrieved_docs = [self.corpus[doc]['text'] for doc in doc_ids]
    #                 shortlisted = reranker_model.reranked_topk(question, retrieved_docs, rerank_k)
    #                 doc_ids = [doc_ids[i] for i in shortlisted]

    #             # Collect the document IDs from the search results
    #             retrieved_doc_ids.append(doc_ids)
                    
    #         # Update the target document with retrieved doc IDs
    #         doc_content['retrieved_doc_ids'] = retrieved_doc_ids

    #         # Save progress
    #         self.save_target_docs()

    def query_target_llm(self, llm, from_ckpt: bool=True):
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.config.attack_config.name}.json'
        with open(output_file, 'r') as f:
            self.target_docs = json.load(f)
            
        for doc_id, doc_content in tqdm(self.target_docs.items(), total=len(self.target_docs), desc="Processing documents"):
            questions = doc_content.get('questions', [])
            retrieved_doc_ids = doc_content.get('retrieved_doc_ids', [])

            llm_responses = doc_content.get('llm_responses', [])
            if not from_ckpt:
                llm_responses = []  # Start fresh if from_ckpt is False
            if len(llm_responses) < len(questions):
                print(f"Reprocessing document: {doc_id}")
                doc_content['llm_responses'] = []  # Clear existing responses to start fresh
                llm_responses = doc_content['llm_responses']

                # Generate responses for all questions
                for i, question in enumerate(questions):
                    # Access the text of each retrieved document directly from self.corpus
                    topk_contents = [self.corpus[doc]['text'][:self.document_slice_size] for doc in retrieved_doc_ids[i]]
                    query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                    try:
                        # Query the LLM for the response
                        max_gen_token_len = self.prompt_len(llm.tokenizer, question, doc_id, i)
                        response = llm.query(query_prompt, max_output_tokens = max_gen_token_len)
                        
                        print(f'Response for doc {doc_id}, question {i}: {response}')
                        llm_responses.append(response)
                    except Exception as e:
                        print(f"Error querying LLM for question {i} in doc {doc_id}: {e}")
                        break  # Stop processing if an error occurs to allow for retry

                print(f"Completed responses for doc {doc_id}: {llm_responses}")
                doc_content['llm_responses'] = llm_responses
                # Save progress after each document is processed
                self.save_target_docs()
            else:
                print(f"All responses already generated for {doc_id}")

        # Final save after processing all documents
        self.save_target_docs()
    
    def calculate_score(self):
        raise NotImplementedError("This method must be implemented in the derived class.")