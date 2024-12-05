
import torch
import random
from tqdm import tqdm
from src.utils import load_json
from src.utils import load_beir_datasets, load_models
from src.prompts import wrap_prompt
import json
import os
import re

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


    def post_filter(self, top_k=3, filtered=True, mode='top'):
        # Iterate over the target documents
        if filtered: # no need to do filter since they are sorted alraedy
            if mode=='top':
                for doc_id, target_info in self.target_docs.items():
                    self.target_docs[doc_id]['questions'] = self.target_docs[doc_id]['questions'][:top_k]
                    self.target_docs[doc_id]['answers'] = self.target_docs[doc_id]['answers'][:top_k]
                    self.target_docs[doc_id]['llm_responses'] = self.target_docs[doc_id]['llm_responses'][:top_k]
                    self.target_docs[doc_id]['retrieved_doc_ids'] = self.target_docs[doc_id]['retrieved_doc_ids'][:top_k]
            elif mode == 'random':
                for doc_id, target_info in self.target_docs.items():
                    # Ensure there are enough elements to sample
                    num_questions = min(top_k, len(self.target_docs[doc_id]['questions']))
                    # Randomly sample top_k items
                    sampled_indices = random.sample(range(len(self.target_docs[doc_id]['questions'])), num_questions)
                    
                    self.target_docs[doc_id]['questions'] = [self.target_docs[doc_id]['questions'][idx] for idx in sampled_indices]
                    self.target_docs[doc_id]['answers'] = [self.target_docs[doc_id]['answers'][idx] for idx in sampled_indices]
                    self.target_docs[doc_id]['llm_responses'] = [self.target_docs[doc_id]['llm_responses'][idx] for idx in sampled_indices]
                    self.target_docs[doc_id]['retrieved_doc_ids'] = [self.target_docs[doc_id]['retrieved_doc_ids'][idx] for idx in sampled_indices]
            
        else:
            for doc_id, target_info in self.target_docs.items():
                questions = target_info.get('questions', [])
                answers = target_info.get('answers', [])
                llm_responses = target_info.get('llm_responses', [])
                retrieved_docs = target_info.get('retrieved_doc_ids', [])
                
                # Ensure we have data to process
                if not questions or not answers or not llm_responses:
                    continue

                # Create a DataFrame with questions for scoring
                df_questions = pd.DataFrame([{
                    'docno': str(doc_id),
                    'text': target_info['text'],
                    'querygen': "\n".join(questions)
                }])

                # Score the questions
                scorer = ElectraScorer()
                query_scorer_pipeline = QueryScorer(scorer)
                scored_df = query_scorer_pipeline(df_questions)
                scored_questions = scored_df.to_dict()
                
                # Retrieve and split the scored questions and their scores
                question_texts = scored_questions['querygen'][0].split('\n')
                score_array = scored_questions['querygen_score'][0]

                # Pair questions with scores and sort them
                paired_questions_scores = sorted(zip(question_texts, score_array), key=lambda x: x[1], reverse=True)
                
                # Select the top-k questions
                top_k_questions = [question for question, _ in paired_questions_scores[:top_k]]
                
                # Find corresponding answers and LLM responses based on the original question order
                question_index_map = {question: idx for idx, question in enumerate(questions)}
                top_k_indices = [question_index_map[q] for q in top_k_questions]

                top_k_answers = [answers[idx] for idx in top_k_indices]
                top_k_llm_responses = [llm_responses[idx] for idx in top_k_indices]
                top_k_retrieved_docs = [retrieved_docs[idx] for idx in top_k_indices]

                # Store the top-k results back in the target_docs dictionary
                self.target_docs[doc_id]['questions'] = top_k_questions
                self.target_docs[doc_id]['answers'] = top_k_answers
                self.target_docs[doc_id]['llm_responses'] = top_k_llm_responses
                self.target_docs[doc_id]['retrieved_doc_ids'] = top_k_retrieved_docs

                print(f"Top-{top_k} Questions for {doc_id}: {top_k_questions}")
                print(f"Corresponding Answers: {top_k_answers}")
                print(f"Corresponding LLM Responses: {top_k_llm_responses}")

        # Save the updated target_docs
        self.save_target_docs()

    # def retrieve_docs_(self, conda_env='colbert', script_path='../ColBERT'):
    #     try:
    #         command = f'conda run --prefix /work/pi_ahoumansadr_umass_edu/yuefeng/conda/envs/{conda_env} python retrieve_for_mia.py --dataset {self.args.eval_dataset} --name {self.args.name} --k {self.args.retrieve_k}'
    #         # Run the command using subprocess and specify the script's working directory via 'cwd'
    #         subprocess.run(command, shell=True, check=True, cwd=script_path)
    #         print("Script executed successfully.")
        
    #     except subprocess.CalledProcessError as e:
    #         print(f"An error occurred: {e}")
    #     except FileNotFoundError as e:
    #         print(f"File not found: {e}")
            
    #     output_dir = 'results/target_docs'
    #     output_file = f'{output_dir}/{self.args.name}.json'
    #     with open(output_file, 'r') as f:
    #         self.target_docs = json.load(f)

    def retrieve_docs_(self, k=5, retriever='colbert'):
        if retriever == 'colbert':
            from ragatouille import RAGPretrainedModel
            import os
            # Load the RAG model with ColBERT index
            index_path = os.path.join("./datasets", self.args.eval_dataset, 'colbert', 'indexes', f"{self.args.eval_dataset}-index")
            model = RAGPretrainedModel.from_index(index_path)
        
            # Iterate over each target doc and retrieve documents for each attack question
            for doc_id, doc_content in self.target_docs.items():
                questions = doc_content.get('questions', [])
                retrieved_doc_ids = []

                print(f"Retrieving documents for doc_id: {doc_id}")

                for question in questions:
                    # Perform the search for each question
                    print(f"Querying for question: {question}")
                    results = model.search(question, k=k)

                    # Collect the document IDs from the search results
                    doc_ids = [result['document_id'] for result in results]
                    retrieved_doc_ids.append(doc_ids)
                    
                # Update the target document with retrieved doc IDs
                doc_content['retrieved_doc_ids'] = retrieved_doc_ids

                # Save progress
                self.save_target_docs()
        
        elif retriever == 'gte':
            import faiss
            import pickle
            from transformers import AutoTokenizer, AutoModel
            import torch.nn.functional as F
            import torch
            import os

            # Paths for FAISS index and model
            dataset = self.args.eval_dataset
            index_folder = os.path.join("./datasets", dataset, "gte", "indexes", f"{dataset}-index")
            faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
            doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

            # Load FAISS index
            print("Loading FAISS index...")
            faiss_index = faiss.read_index(faiss_index_path)

            # Load document IDs
            print("Loading document IDs...")
            with open(doc_ids_path, "rb") as f:
                doc_ids = pickle.load(f)

            # Load GTE model
            model_name = "Alibaba-NLP/gte-large-en-v1.5"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to("cuda")
            model.eval()

            # Function for encoding queries
            def encode_query(query):
                inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs)
                    query_embedding = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)  # CLS pooling and normalization
                return query_embedding.cpu().numpy()

            # Iterate over each target doc and retrieve documents for each attack question
            for doc_id, doc_content in self.target_docs.items():
                questions = doc_content.get('questions', [])
                retrieved_doc_ids = []

                print(f"Retrieving documents for doc_id: {doc_id} using GTE")

                for question in questions:
    
                    query_embedding = encode_query(question)
                    distances, indices = faiss_index.search(query_embedding, k)
                    doc_ids_for_query = [doc_ids[i] for i in indices[0]]
                    retrieved_doc_ids.append(doc_ids_for_query)
                    print(f"Retrieved doc IDs for question '{question}': {doc_ids_for_query}")

                # Update the target document with retrieved doc IDs
                doc_content['retrieved_doc_ids'] = retrieved_doc_ids

                # Save progress
                self.save_target_docs()
        
        elif retriever == 'bge':
            from transformers import AutoTokenizer, AutoModel
            import faiss
            import pickle
            import torch
            import os

            model_name = "BAAI/bge-large-en-v1.5"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to('cuda').eval()

            # Paths for FAISS index and document IDs
            dataset = self.args.eval_dataset
            index_folder = os.path.join("./datasets", dataset, "bge", "indexes", f"{dataset}-index")
            faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
            doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

            # Load FAISS index
            print(f"Loading FAISS index from: {faiss_index_path}")
            faiss_index = faiss.read_index(faiss_index_path)

            # Load document IDs
            print(f"Loading document IDs from: {doc_ids_path}")
            with open(doc_ids_path, "rb") as f:
                doc_ids = pickle.load(f)

            # Function for embedding normalization
            def normalize_embeddings(embeddings):
                return torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Query encoding function
            def encode_query(query):
                inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs)
                    query_embedding = normalize_embeddings(outputs.last_hidden_state[:, 0])  # CLS pooling
                return query_embedding.cpu().numpy()

            # Iterate over each target doc and retrieve documents for each attack question
            print("Retrieving documents using BGE...")
            for doc_id, doc_content in self.target_docs.items():
                questions = doc_content.get('questions', [])
                retrieved_doc_ids = []

                print(f"Retrieving documents for doc_id: {doc_id}")

                for question in questions:
                    try:
                        # Encode the query
                        query_embedding = encode_query(question)

                        # Perform FAISS search
                        distances, indices = faiss_index.search(query_embedding, k)
                        # Map indices to document IDs
                        retrieved_ids = [doc_ids[i] for i in indices[0]]
                        retrieved_doc_ids.append(retrieved_ids)
                        print(f"Retrieved document IDs for question '{question}': {retrieved_ids}")

                    except Exception as e:
                        print(f"Error during retrieval for question '{question}' in doc {doc_id}: {e}")
                        continue

                # Update the target document with retrieved document IDs
                doc_content['retrieved_doc_ids'] = retrieved_doc_ids
                # Save progress
                self.save_target_docs()

        elif retriever == 'contriever':
            # Import necessary libraries
            import os
            import faiss
            import pickle
            import torch
            from transformers import AutoTokenizer, AutoModel

            model_name = "facebook/contriever"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to('cuda').eval()

            # Paths for FAISS index and document IDs
            dataset = self.args.eval_dataset
            index_folder = os.path.join("./datasets", dataset, 'contriever', 'indexes', f"{dataset}-index")
            faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
            doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

            # Load FAISS index
            print(f"Loading FAISS index from {faiss_index_path}")
            faiss_index = faiss.read_index(faiss_index_path)

            # Load document IDs
            print(f"Loading document IDs from {doc_ids_path}")
            with open(doc_ids_path, "rb") as f:
                doc_ids = pickle.load(f)

            # Mean pooling function for query encoding
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings

            # Query encoding function
            def encode_query(query, tokenizer, model, device):
                inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    query_embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
                return query_embedding.cpu().numpy()

            # Iterate over each target document and retrieve results for its questions
            print("Retrieving documents using Contriever...")
            for doc_id, doc_content in self.target_docs.items():
                questions = doc_content.get('questions', [])
                retrieved_doc_ids = []

                for question in questions:
                    try:
                        query_embedding = encode_query(question, self.tokenizer, self.model, torch.device("cuda"))
                        distances, indices = faiss_index.search(query_embedding, k)
                        retrieved_ids = [doc_ids[i] for i in indices[0]]
                        retrieved_doc_ids.append(retrieved_ids)
                        print(f"Retrieved document IDs for question '{question}': {retrieved_ids}")

                    except Exception as e:
                        print(f"Error during retrieval for question '{question}' in doc {doc_id}: {e}")
                        continue

                # Update the target document with retrieved document IDs
                doc_content['retrieved_doc_ids'] = retrieved_doc_ids

                # Save progress after each document
                self.save_target_docs()

        else:
            raise ValueError(f"Retriever '{retriever}' is not supported.")

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
                    answer_prompt = wrap_prompt(question, [target_doc], prompt_id=4, atk=True)
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
        
        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'
        with open(output_file, 'r') as f:
            self.target_docs = json.load(f)
            
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
                    topk_contents = [self.corpus[doc]['text'][:2048] for doc in retrieved_doc_ids[i]]
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



