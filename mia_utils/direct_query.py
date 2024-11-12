from sentence_transformers import SentenceTransformer
import torch
import random
from tqdm import tqdm
from src.utils import load_json
from src.utils import load_beir_datasets, load_models
from src.prompts import wrap_prompt
import json
import os
import re

from beir.retrieval.search.dense import DenseRetrievalFaissSearch as DRFS
from beir.retrieval import models
import faiss
import subprocess

import pandas as pd
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
import re

class Direct_Query_Attacker():
    def __init__(self, args, **kwargs) -> None:

        self.args = args
        self.attack_method = args.attack_method
        
        # reserved for retriever
        self.model = kwargs.get('model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)

        #yf
        self.shadow_llm = kwargs.get('shadow_llm', None)
        self.target_docs = kwargs.get('target_docs', None)

        #load dataset
        self.corpus = kwargs.get('corpus', None)

    def save_target_docs(self):
        # Define the output directory and filename dynamically
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Save the target_docs dictionary to the specified JSON file
        with open(output_file, 'w') as f:
            json.dump(self.target_docs, f, indent=4)

    def generate_questions(self):
        for doc_id, target_info in self.target_docs.items():
            target_doc = target_info['text']
            query = f'Does this:\n\"{target_doc}\n"appear in the context?'
            self.target_docs[doc_id]['questions'] = [query]
            # Save the updated target_docs
        self.save_target_docs()

    def retrieve_docs_(self, conda_env='colbert', script_path='../ColBERT'):
        try:
            command = f'conda run --prefix /work/pi_ahoumansadr_umass_edu/yuefeng/conda/envs/{conda_env} python retrieve_for_mia.py --dataset {self.args.eval_dataset} --name {self.args.name}'
            # Run the command using subprocess and specify the script's working directory via 'cwd'
            subprocess.run(command, shell=True, check=True, cwd=script_path)
            print("Script executed successfully.")
        
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'
        with open(output_file, 'r') as f:
            self.target_docs = json.load(f)


    def query_target_llm(self, llm, from_ckpt=True):

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'
        with open(output_file, 'r') as f:
            self.target_docs = json.load(f)

        for doc_id, doc_content in self.target_docs.items():
            questions = doc_content.get('questions', [])
            retrieved_doc_ids = doc_content.get('retrieved_doc_ids', [])

            llm_responses = doc_content.get('llm_responses', [])
            print(doc_content)
            if not from_ckpt:
                llm_responses = []  # Start fresh if from_ckpt is False
            if len(llm_responses) < len(questions):
                print(f"Reprocessing document: {doc_id}")
                doc_content['llm_responses'] = []  # Clear existing responses to start fresh
                llm_responses = doc_content['llm_responses']

                # Generate responses for all questions
                for i, question in enumerate(questions):
                    # Access the text of each retrieved document directly from self.corpus
                    topk_contents = [self.corpus[doc]['text'] for doc in retrieved_doc_ids[i]]
                    query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                    try:
                        # Query the LLM for the response
                        response = llm.query(query_prompt)
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

    def calculate_accuracy_(self):
        def extract_yes_no(answer):
            """Extract 'Yes' or 'No' from a response."""
            match = re.search(r'\b(Yes|No)\b', answer, re.IGNORECASE)
            return match.group(1).capitalize() if match else "Unknown"

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'
        with open(output_file, 'r') as f:
            data = json.load(f)

        all_count = 0
        correct_count = 0
        for doc_id, doc_data in data.items():
            
            expected_answer = "Yes" if doc_data.get("mem", "").lower() == "yes" else "No"
            llm_response = extract_yes_no(item.get("llm_response", "")[0])
            if expected_answer == llm_response:
                correct_count += 1
            all_count += 1
            # Calculate accuracy for the document
            accuracy = correct_count / all_count * 100 if all_count > 0 else 0
            print(f"Until Document {doc_id} - Accuracy: {accuracy:.2f}%")




