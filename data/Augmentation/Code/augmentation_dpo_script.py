import time

import pandas as pd
import re
from huggingface_hub import InferenceClient


def generate_seed(k, variant_num):
    # A base_seed to make sure all seeds are generated based on some constant
    base_seed = 42
    return base_seed * variant_num + k


def extract_augmented_sentence(text):
    match = re.search(r"\"(.*?)\"", text)
    if match:
        return match.group(1)
    else:
        return None


def augment_with_LLM(sentence, seed, attempt=0, max_attempts=5):
    if attempt >= max_attempts:
        print(f"Failed to generate an augmented version for: {sentence}. Exceeded {max_attempts} attempts.")
        return None

    prompt = ("""You are an advanced language model specialized in rephrasing sentences while retaining their original
              meaning and context.

              Your task is to generate a new version of the given sentence, ensuring that the meaning remains the same.

              Example: 
              Original: "We collect the following personal information: name, postal address, and phone number."
              Rephrased: "We acquire the listed personal data: name, postal address, and telephone number."

              Original: "{sentence}"
              Rephrased: """)

    while True:
        try:
            generated_text = client.text_generation(prompt.format(sentence=sentence), temperature=0.9,
                                                    max_new_tokens=128,
                                                    seed=seed)
            generated_sentence = extract_augmented_sentence(generated_text)

            if generated_sentence is None:
                return augment_with_LLM(sentence, seed + 1, attempt + 1, max_attempts)

            return generated_sentence

        except Exception as e:
            print(f"An exception occurred: {e}")
            time.sleep(5)  # Wait for 5 seconds before trying again
            print("Retrying...")


token = "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"
client = InferenceClient(token=token, model="meta-llama/Llama-2-70b-chat-hf")

data_df = pd.read_csv('../../training_data_dpo.csv')

# Separate the minority class
df_minority = data_df[data_df['DPO'] == 1]

# Specify the number of paraphrases for each sentence
num_variants = 5
augmented_rows = []
total = len(df_minority)

# Generate paraphrases for the minority class to balance the dataset
for i, (idx, row) in enumerate(df_minority.iterrows()):
    sentence = row['Text']
    row['Text'] = row['Text'].replace('"', '')
    print(f"Original: {sentence}")

    for variant in range(num_variants):
        augmented_sentence = augment_with_LLM(sentence, generate_seed(idx, variant))

        if augmented_sentence:
            print(f"Rephrased (Version {variant + 1}): {augmented_sentence}")
            augmented_rows.append([augmented_sentence, 1])

    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{total} ({((i + 1) / total) * 100:.2f}%)")
    print("------------------------")  # Separator for better visual clarity

df_augmented = pd.DataFrame(augmented_rows, columns=['Text', 'DPO'])

# Save the augmented DataFrame
output_path = '../augmented_dpo_compliant_5.csv'
df_augmented.to_csv(output_path, index=False)
