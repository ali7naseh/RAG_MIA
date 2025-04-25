
from mia_utils.base import BaseAttacker
import json
import numpy as np
from tqdm import tqdm
import re
import torch as ch
from typing import List

import string
import nltk
from nltk.corpus import stopwords

from collections import defaultdict

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def normalize(word):
    """
    Normalize a word by removing punctuation, converting to lowercase, and stripping spaces.
    Args:
        word (str): The input word to normalize.
    Returns:
        str: The normalized word.
    """
    return word.strip().lower().translate(str.maketrans('', '', string.punctuation))


class MBA_Attacker(BaseAttacker):
    """
        MBA attack, as described in the paper 'Mask-based Membership Inference Attacks for Retrieval-Augmented Generation'
    """
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.num_masks = kwargs.get('num_masks', 10)
        if self.config.attack_config.proxy_lm_mba is None:
            raise ValueError("Please specify the proxy LM model for the MBA attack.")

        # MBA-specific models
        if kwargs.get('load_misc_models', True):
            
            # Spell-correction model
            self.spell_correction = pipeline("text2text-generation",
                                             model="oliverguhr/spelling-correction-english-base",
                                             device='cuda')

            # Proxy LM
            # Original paper used gpt-xl
            proxy_lm = self.config.attack_config.proxy_lm_mba #"meta-llama/Llama-3.2-3B"
            self.proxy_lm_tokenizer = AutoTokenizer.from_pretrained(proxy_lm)
            self.proxy_lm_model = AutoModelForCausalLM.from_pretrained(proxy_lm).eval().cuda()

        # Ensure NLTK stopwords are downloaded
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        # Add custom abbreviations (include <human> and <bot> tags that are part of HealthChat dataset)
        self.stop_words.update(["etc", "etc.", "e.g.", "i.e.", "i.e", "et al.", "<human>:", "<bot>:"])

        self.punctuations = set(string.punctuation)
    
    def free_space(self):
        """
            Delete proxy-LM, spell-correction, etc.
        """
        if hasattr(self, 'spell_correction'):
            del self.spell_correction
        if hasattr(self, 'proxy_lm_tokenizer'):
            del self.proxy_lm_tokenizer
        if hasattr(self, 'proxy_lm_model'):
            del self.proxy_lm_model
        # Free up GPU memory
        ch.cuda.empty_cache()

    def prompt_len(self, tokenizer, question: str, doc_id: int, question_id: int):
        # Look at number of tokens in answer
        answer_dict_str = "\n".join([f"[MASK_{k}]: v" for k, v in self.target_docs[doc_id]['answers'].items()])
        num_tokens_answer_dict_str = len(tokenizer.tokenize(answer_dict_str))
        # Add a buffer of 12 tokens
        return num_tokens_answer_dict_str + 12

    def _fragmented_word_extraction(self, words: List[str]) -> List[int]:
        split_word_indices = []
        for i, word in enumerate(words):
            tokenized_word = self.proxy_lm_tokenizer(word)
            if len(tokenized_word['input_ids']) > 2: # Account for BOS token
                split_word_indices.append(i)
        return split_word_indices

    def corrected_word_extraction(self,
                                  words: List[str],
                                  fragmented_indices: List[int]):
        corrected_fragmented_words = defaultdict()
        for i in fragmented_indices:
            # Create sub-sentence with the fragmented word and 2 words before it
            sub_sentence = " ".join(words[max(0, i-2):i+1])
            corrected_response = self.spell_correction(sub_sentence)[0]['generated_text']
            # Remove any period added at the end that wasn't already there
            if sub_sentence[-1] != ".":
                corrected_response = corrected_response.rstrip(".")
            # Get last corrected word
            corrected_word = corrected_response.split()[-1]
            # If word remained unchanged, use original word
            if corrected_word.lower() == words[i].lower():
                corrected_word = words[i]

            corrected_fragmented_words[i] = corrected_word
        return corrected_fragmented_words

    @ch.no_grad()
    def compute_rank(self, prefix: str, token: int):
        # Given prefix, compute rank(P(token|prefix))
        input = self.proxy_lm_tokenizer(prefix, return_tensors="pt", truncation=True)
        input = {k: v.cuda() for k, v in input.items()}
        logits = self.proxy_lm_model(**input).logits
        # Compute 'rank' of token (where higher rank means higher probability)
        # Can use argsort and look for the index of the token
        logits_argsort = ch.argsort(logits[0, -1], descending=True).cpu().numpy()
        # print(logits_argsort[:25])
        rank = np.where(logits_argsort == token)[0][0]
        # print("First 25 candidates were:", ",".join([self.proxy_lm_tokenizer.decode([x]) for x in logits_argsort[:25]]))
        return rank

    def mba_pipeline(self, document: str, M: int):
        # Extracted fragmented words using proxy_lm tokenizer
        document_words = document.split()
        fragmented_word_indices = self._fragmented_word_extraction(document_words)
        # Get corrected words for the fragmented words
        corrected_word_extraction = self.corrected_word_extraction(document_words, fragmented_word_indices)
        
        # Divide document_words into M fragments
        fragment_size = int(np.ceil(len(document_words) / M))
        masked = np.zeros(len(document_words), dtype=bool)
        mask_answers = []
        prefix = ""
        for i in range(0, len(document_words), fragment_size):
            mask_scores_within = []
            masked_words_within = []

            # We want one mask word per fragment
            for j in range(i, i + fragment_size):
                # Break if we reach the end of the document
                if j >= len(document_words):
                    break

                # Skip the first 5 words (source: authors)
                if j < 5:
                    score = -1
                    masked_words_within.append([""])
                # If stop-word or punctuation, assign score of -1
                elif document_words[j].lower() in self.stop_words or document_words[j] in self.punctuations:
                    score = -1
                    masked_words_within.append([""])
                # If right next to an already-masked word, do not mask
                elif j > 0 and masked[j-1]:
                    score = -1
                    masked_words_within.append([""])
                else:
                    if j not in fragmented_word_indices:
                        # print("NOT-FRAGMENT")
                        tokenized_word = self.proxy_lm_tokenizer(document_words[j])['input_ids']
                        # Skip BOS token
                        # tokenized_word = tokenized_word[1:]
                        # Word not fragmented, can compute score directly
                        score = self.compute_rank(prefix + " ", tokenized_word[0])
                        masked_words_within.append([document_words[j]])
                        # print(f"Rank is {score} for [{self.proxy_lm_tokenizer.decode(tokenized_word)}] ({tokenized_word[0]}) given: {prefix}")
                    else:
                        # print("FRAGMENT")
                        tokenized_corrected_word = self.proxy_lm_tokenizer(corrected_word_extraction[j])['input_ids']
                        # Skip BOS token
                        # tokenized_corrected_word = tokenized_corrected_word[1:]
                        # Word is fragmented, compute score for each possible token
                        scores_ = []
                        toks_inner = []

                        for tok in tokenized_corrected_word:
                            toks_inner_str = self.proxy_lm_tokenizer.decode(toks_inner)
                            prefix_inner = prefix + " " + toks_inner_str
                            score = self.compute_rank(prefix_inner, tok)
                            # print(f"Rank is {score} for [{self.proxy_lm_tokenizer.decode([tok])}] ({tok}) given: {prefix_inner}")
                            scores_.append(score)
                            toks_inner.append(tok)
                        score = max(scores_)
                        # print(f"Rank of {corrected_word_extraction[j]} is {score}")
                        masked_words_within.append([corrected_word_extraction[j], document_words[j]])

                # Track score
                mask_scores_within.append(score)

                # Track prefix
                if len(prefix) == 0:
                    prefix = document_words[j]
                else:
                    prefix += " " + document_words[j]
            
            # Select argmax to mask
            mask_idx = np.argmax(mask_scores_within)
            masked[i + mask_idx] = True
            mask_answers.append(masked_words_within[mask_idx])
            # print("#" * 30)
        
        mask_answers = {i+1: mask_answers[i] for i in range(len(mask_answers))}

        # Now need to generate masked document where these words are replaced with [Mask_1] etc.
        j = 1
        masked_document = []
        for i, word in enumerate(document_words):
            if masked[i]:
                masked_document.append(f"[MASK_{j}]")
                j += 1
            else:
                masked_document.append(word)
        
        masked_document = " ".join(masked_document)
        # print(f"Masked document: {masked_document}")
        # print(f"Mask answers: {mask_answers}")
        # print("\n\n")
        return masked_document, mask_answers

    def generate_questions(self):
        template = ('Predict the answers of the masks in the given texts:\n\n{masked_document}\n\n'
                    'You should strictly follow the contexts below to answer, and provide your answers in the specified format:\n'
                    '"[Mask_i]: answer_i\n" for each mask:')

        for doc_id, target_info in tqdm(self.target_docs.items(), desc="Generating Questions", total=len(self.target_docs)):
            target_doc = target_info['text']

            # Use version of target_doc that is at most self.document_slice_size characters long
            # Generated masked document (and answers, to be used later)
            masked_document, mask_answers = self.mba_pipeline(target_doc[:self.document_slice_size], self.num_masks)
            query = template.format(masked_document=masked_document)

            self.target_docs[doc_id]['questions'] = [query]
            self.target_docs[doc_id]['answers'] = mask_answers
            # Save the updated target_docs
        self.save_target_docs()

    def calculate_score(self):
        def extract_mask_answers(text):
            # Regular expression to match the mask pattern
            pattern = r'\[?MASK?_?(\d+)\]?:?\s*\**\s*([\w\-\' ]+)'
    
            # Find all matches
            matches = re.findall(pattern, text, re.IGNORECASE)
    
            # Create a dictionary from the matches
            mask_mappings = {id: answer.strip() for id, answer in matches}
    
            return mask_mappings

        output_dir = 'results/target_docs'
        output_file = f'{output_dir}/{self.config.attack_config.name}.json'
        with open(output_file, 'r') as f:
            data = json.load(f)

        for doc_id, doc_data in data.items():

            correct_answer = doc_data['answers']
            llm_response = doc_data.get("llm_responses", [""])[0]

            # Compare LLM response with correct answers
            llm_answer = extract_mask_answers(llm_response)
            total_questions = len(correct_answer)

            # Count number of positions where the LLM response matches the correct answer
            correct_count = 0
            for mask, answer in correct_answer.items():
                normalized_answer = normalize(llm_answer.get(mask, ''))
                if normalized_answer in [normalize(ans) for ans in answer]:
                    correct_count += 1

            # Calculate accuracy for the document
            accuracy = correct_count / total_questions * 100
            print(f"Document {doc_id} - Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")

            # Save the accuracy back to the dict
            self.target_docs[doc_id]['accuracy'] = accuracy

        # Save the updated target_docs
        self.save_target_docs()
