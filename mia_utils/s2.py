
from src.prompts import wrap_prompt
from mia_utils.base import BaseAttacker
import json
import os

from nltk.translate.bleu_score import sentence_bleu


class S2_Attacker(BaseAttacker):
    """
        S2-MIA, from the work 'Generating Is Believing: Membership Inference Attacks against Retrieval-Augmented Generation'
    """
    def generate_questions(self):
        """
        Generate queries for each target document by splitting the text into query and remaining parts.
        """
        for doc_id, target_info in self.target_docs.items():
            target_doc = target_info['text']

            # Split the document into two parts: first half as query, second half as remaining
            split_idx = len(target_doc) // 2
            query_text = target_doc[:split_idx].strip()  # First half
            remaining_text = target_doc[split_idx:].strip()  # Second half

            # Construct the query in the specified format
            formatted_query = (
                f"Given the [Query]: \"{query_text}\".\n"
                f"Do not include any introductory or explanatory text, use the following format for output: "
                f"{{[Response]: 'Provide a concise response directly addressing the [Query] by using the most relevant and matching text in the prompt.'}}"
            )

            # Update the document's metadata
            self.target_docs[doc_id]['questions'] = [formatted_query]

        # Save the updated target_docs
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
                    topk_contents = [self.corpus[doc]['text'] for doc in retrieved_doc_ids[i]]
                    query_prompt = wrap_prompt(question, topk_contents, prompt_id=4)

                    try:
                        # Query the LLM for the response
                        max_gen_token_len = len(llm.tokenizer.tokenize(question))
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
        """
        Calculate BLEU score for each document.
        """
        def extract_response(answer):
            """Extract the generated response from the LLM's output."""
            return answer.strip()

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.args.name}.json'

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file {output_file} does not exist.")

        with open(output_file, 'r') as f:
            data = json.load(f)

        for doc_id, doc_data in data.items():
            # Retrieve the target text and LLM's generated text
            target_text = doc_data.get("text", "")
            generated_text = extract_response(doc_data.get("llm_responses", [""])[0])

            print(generated_text)
            # Calculate BLEU score
            reference = [target_text.split()]  # Tokenize target text as a reference
            candidate = generated_text.split()  # Tokenize generated text
            print(reference)
            print(candidate)
            bleu_score = sentence_bleu(reference, candidate)

            self.target_docs[doc_id]['bleu_score'] = bleu_score
            print(f"Document {doc_id} - BLEU Score: {bleu_score:.4f}")

        # Save the updated target_docs
        self.save_target_docs()
