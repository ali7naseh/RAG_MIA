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

class MIA_Attacker():
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

    def generate_questions_GPT_4(self, source_docs):
        # Iterate over the target documents
        for doc_id, target_info in self.target_docs.items():
            source_info = source_docs.get(doc_id, {})
            questions = source_info.get('questions', [])
            combined_questions = "\n".join(questions)
            self.target_docs[doc_id]['all_questions'] = combined_questions
            print(f"Combined questions for {doc_id}: {combined_questions}")
        print('All questions are generated.')
        # Save the updated target_docs
        self.save_target_docs()

    def generate_questions_llama3(self):
        question_pattern = r"Q\d+:\s*(.*?)$"  # Regex pattern to match "Q1: xxx" format for questions
        for doc_id, target_info in self.target_docs.items():
            target_doc = target_info['text']
            gen_prompt = (f"Generate exactly {self.args.N} Yes/No questions based on the following document.\n"
                        f"Document: {target_doc}\n"
                        f"Each question should:\n"
                        f"- Be meaningful and understandable on its own.\n"
                        f"- Include exact phrases, names, numbers, or unique facts from the document.\n"
                        f"- Use wording directly from the document where possible.\n"
                        f"- Be specific enough to retrieve this document when used as a query.\n"
                        f"Avoid general or vague questions that could apply to other documents.\n"
                        f"Format your questions as follows:\n"
                        f"Q1: ...\nQ2: ...\n... up to Q{self.args.N}.")

            # Get the response from the LLM for questions
            response = self.shadow_llm.query(gen_prompt)

            # Use regex to extract questions formatted as Q1: xxx, Q2: xxx, etc.
            questions = re.findall(question_pattern, response, re.MULTILINE)
            cleaned_questions = [q.strip() for q in questions if q.strip()]
            combined_questions = "\n".join(cleaned_questions)
            self.target_docs[doc_id]['all_questions'] = combined_questions

        # Save the updated target_docs
        self.save_target_docs()

    def generate_questions_D2Q_IR(self):
        doc2query = Doc2Query(append=False, num_samples=self.args.N)  # Set the number of samples/questions to generate
        documents = [{'docno': str(doc_id), 'text': doc_data['text']} for doc_id, doc_data in self.target_docs.items()]
        df = pd.DataFrame(documents)

        # Generate queries using the doc2query pipeline
        result = doc2query(df)
        result_dict = result.to_dict()

        for idx, doc_id in enumerate(result_dict['docno'].values()):
            doc_id = str(doc_id)
            if doc_id in self.target_docs:
                generated_queries = result_dict.get('querygen', {}).get(idx, '').split('\n')
                cleaned_queries = [q.strip() + '?' for q in generated_queries if q.strip()]
                combined_questions = "\n".join(cleaned_queries)
                # Store combined queries in the 'all_questions' field
                self.target_docs[doc_id]['all_questions'] = combined_questions
                print(f"Generated Queries for {doc_id}: {cleaned_queries}")
        # Save the updated target_docs
        self.save_target_docs()

    def filter_questions_topk(self, top_k=3):
        # Iterate over the target documents
        for doc_id, target_info in self.target_docs.items():
            all_questions = target_info.get('all_questions', None)
            if not all_questions:
                continue

            question_list = all_questions.split('\n')
            df_questions = pd.DataFrame([{
                'docno': str(doc_id),
                'text': target_info['text'],
                'querygen': "\n".join(question_list)
            }])
            scorer = ElectraScorer()
            query_scorer_pipeline = QueryScorer(scorer)
            scored_df = query_scorer_pipeline(df_questions)
            scored_questions = scored_df.to_dict()
            question_texts = scored_questions['querygen'][0].split('\n')
            score_array = scored_questions['querygen_score'][0]
            paired_questions_scores = sorted(zip(question_texts, score_array), key=lambda x: x[1], reverse=True)
            top_k_questions = [question for question, _ in paired_questions_scores[:top_k]]

            # Store the top-k questions back in the 'questions' field
            self.target_docs[doc_id]['questions'] = top_k_questions
            print(f"Top-{top_k} Questions for {doc_id}: {top_k_questions}")

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

    def generate_ground_truth_answers(self, llm=None,  from_ckpt=True):
        answer_pattern = r"A\d+:\s*(Yes|No)"  # Regex pattern to match "A1: Yes/No" format for answers

        # Iterate over each target document in the list
        for q, doc_info in self.target_docs.items():
            questions = doc_info.get('questions', [])
            target_doc = doc_info['text']

            answers = doc_info.get('answers', [])
            if not from_ckpt:
                answers=[]
            if len(answers) < len(questions):
                print(f"Reprocessing document: {q}")
                doc_info['answers'] = []  # Clear existing answers to start fresh
                answers = doc_info['answers']

                # Generate answers for all questions
                for idx, question in enumerate(questions, start=1):
                    answer_prompt = wrap_prompt(question, [target_doc], prompt_id=4)
                    try:
                        answer_response = llm.query(answer_prompt)
                        print('answer response: ', answer_response)
                        answers.append(answer_response)
                        doc_info['answers'] = answers
                    except Exception as e:
                        print(f"Error generating answer for question {idx} in doc {q}: {e}")
                        break  # Stop processing if an error occurs to allow for retry
                print(f"Answers for {q}: {answers}")
                # Save progress after each answer
                self.save_target_docs()
            else:
                print(f"All answers already generated for {q}")
        # Save final state after processing all documents
        self.save_target_docs()

    def query_target_llm(self, llm, from_ckpt=True):
        for doc_id, doc_content in self.target_docs.items():
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

        for doc_id, doc_data in data.items():

            # Skip the document if it doesn't meet the required number of questions or responses
            # if len(doc_data['answers']) < self.args.N or len(doc_data['questions']) < self.args.N:
            #     print(f"Skipping Document {doc_id}: Not enough questions or responses")
            #     continue

            correct_answers = doc_data['answers']
            llm_responses = doc_data['llm_responses']
            total_questions = len(correct_answers)
            correct_count = 0

            # Compare LLM responses with correct answers
            for i in range(total_questions):
                # Extract 'Yes' or 'No' from both the LLM response and the correct answer
                correct_answer = extract_yes_no(correct_answers[i])
                llm_answer = extract_yes_no(llm_responses[i])

                if correct_answer == llm_answer:
                    correct_count += 1

            # Calculate accuracy for the document
            accuracy = correct_count / total_questions * 100
            print(f"Document {doc_id} - Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")

            # Save the accuracy back to the dict
            self.target_docs[doc_id]['accuracy'] = accuracy



