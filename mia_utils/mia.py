
from mia_utils.base import BaseAttacker
import random
from src.prompts import wrap_prompt
import json
import re

import pandas as pd
from pyterrier_doc2query import Doc2Query, QueryScorer
from pyterrier_dr import ElectraScorer
import re


class MIA_Attacker(BaseAttacker):
    def generate_questions_GPT_4(self, source_docs):
        # Iterate over the target documents
        for doc_id, target_info in self.target_docs.items():
            source_info = source_docs.get(doc_id, {})
            questions = source_info.get('questions', [])
            combined_questions = "\n".join(questions)
            self.target_docs[doc_id]['all_questions'] = combined_questions
            self.target_docs[doc_id]['questions'] = questions
            print(f"Combined questions for {doc_id}: {combined_questions}")
        print('All questions are generated.')
        # Save the updated target_docs
        self.save_target_docs()

    def generate_questions_llama3(self):
        question_pattern = r"Q\d+:\s*(.*?)$"  # Regex pattern to match "Q1: xxx" format for questions
        for doc_id, target_info in self.target_docs.items():
            target_doc = target_info['text']
            gen_prompt = (f"Generate exactly {self.config.attack_config.N} Yes/No questions based on the following document.\n"
                        f"Document: {target_doc}\n"
                        f"Each question should:\n"
                        f"- Be meaningful and understandable on its own.\n"
                        f"- Include exact phrases, names, numbers, or unique facts from the document.\n"
                        f"- Use wording directly from the document where possible.\n"
                        f"- Be specific enough to retrieve this document when used as a query.\n"
                        f"Avoid general or vague questions that could apply to other documents.\n"
                        f"Format your questions as follows:\n"
                        f"Q1: ...\nQ2: ...\n... up to Q{self.config.attack_config.N}.")

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
        doc2query = Doc2Query(append=False, num_samples=self.config.attack_config.N)  # Set the number of samples/questions to generate
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

    def prompt_len(self, tokenizer, question: str, doc_id: int, question_id: int):
        # We only expect yes/no/I don't know in responses
        return 5

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
            scorer = ElectraScorer(verbose=False)
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

    def post_filter(self,
                    top_k: int=3,
                    filtered: bool=True,
                    mode: str='top'):
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
                scorer = ElectraScorer(verbose=False)
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

    def generate_ground_truth_answers(self, llm=None, from_ckpt: bool=True):
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
                    answer_prompt = wrap_prompt(question, [target_doc], prompt_id=4, atk=True,
                                                context_free_response=self.config.rag_config.context_free_response)
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

    def calculate_score(self):
        def extract_yes_no(answer):
            """Extract 'Yes' or 'No' from a response."""
            match = re.search(r'\b(Yes|No)\b', answer, re.IGNORECASE)
            return match.group(1).capitalize() if match else "Unknown"

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.config.attack_config.name}.json'
        with open(output_file, 'r') as f:
            data = json.load(f)

        for doc_id, doc_data in data.items():

            # Skip the document if it doesn't meet the required number of questions or responses
            # if len(doc_data['answers']) < self.config.attack_config.N or len(doc_data['questions']) < self.config.attack_config.N:
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
