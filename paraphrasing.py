from openai import OpenAI

from tqdm import tqdm
import json
import spacy
import re

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_lg")


def main(key, filepath):
    client = OpenAI(api_key=key)

    # Open and read the JSON file
    with open(filepath, 'r') as file:
        corpus = json.load(file)
    
    def paraphrase_text_v2(input_text):
        # Define the new paraphrasing prompt
        prompt = (
            "As an expert copy-editor, please rewrite the following text in <paraphrase> tags in your own voice while ensuring "
            "that the final output contains the same information as the original text and has roughly the "
            "same length. Please paraphrase all sentences and do not omit any crucial details\n\n"
            f"Here is the input text: <paraphrase>\n{input_text}\n</paraphrase>"
        )
    
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
    
        # Extract and return the paraphrased text
        paraphrased_text = completion.choices[0].message.content.strip()
        # Extract the paraphrased text
        if "<paraphrase>" in paraphrased_text and "</paraphrase>" in paraphrased_text:
            paraphrased_text_regex = re.search(r'<paraphrase>(.*?)</paraphrase>', paraphrased_text, re.DOTALL)
            if paraphrased_text_regex is None:
                print(f"Failed to extract paraphrased text for input:\n\n{paraphrased_text}")
                return paraphrased_text
            paraphrased_text = paraphrased_text_regex.group(1).strip()
    
        return paraphrased_text

    for key, sample in tqdm(corpus.items(), desc="Processing samples"):
        # If original_questions key contained, it was already processed
        if 'original_questions' in sample:
            continue

        questions = sample['questions']

        # Generate paraphrased text using GPT-4o
        para_questions = []
        for question in questions:
            # par_ques = paraphrase_text_v1(question, temperature=TEMPERATURE)
            par_ques = paraphrase_text_v2(question)
            para_questions.append(par_ques)

        # Update the sample with the paraphrased text
        sample['questions'] = para_questions
        # Keep track of original questions also
        sample['original_questions'] = questions
    
        # Save the dictionary as a JSON file
        with open(filepath, 'w') as json_file:
            json.dump(corpus, json_file, indent=4)


if __name__ == "__main__":
    import sys
    key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
    filepath = sys.argv[1]
    main(key, filepath)
