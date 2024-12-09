
from mia_utils.base import BaseAttacker
import json
import re

import re

class Direct_Query_Attacker(BaseAttacker):
    """
        Attack described in the work 'Is My Data in Your Retrieval Database? Membership Inference Attacks Against Retrieval Augmented Generation'
    """
    def generate_questions(self):
        for doc_id, target_info in self.target_docs.items():
            target_doc = target_info['text']
            query = f'Does this:\n\"{target_doc}\n"appear in the context? Answer with Yes or No'
            self.target_docs[doc_id]['questions'] = [query]
            # Save the updated target_docs
        self.save_target_docs()

    def calculate_score(self):
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
