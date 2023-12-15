import pandas as pd
from classification_function import evaluate_model

# Load data
data_df = pd.read_csv('../data/GDPR.csv', sep='\t')

# Prompt setup
prompt1 = """You are a compliance officer specialized in checking sentences for GDPR compliance. 

Your task is to analyze the content of the provided sentence and determine whether it conforms to the Data Protection Officer (DPO) requirement.

Assign a binary score to each sentence:
- 0 if the sentence does not contain relevant DPO-related information.
- 1 if the sentence discloses DPO-relevant information and is compliant to the GDPR requirement.

Consider a sentence compliant (assign label '1') if it either names the Data Protection Officer (DPO) or an 
equivalent authority, or provides their contact details. 

Note: Sentences are also compliant if they refer to some list or enumeration of the required information. 
For example, the sentence 'You can reach our data protection officer via:' is considered compliant, even though the sentence itself does not contain the contact details.

For confidentiality and privacy reasons, the sentences have been anonymized, i.e., numeric values have been randomized,
and names, email addresses, companies and URLs have been substituted with generic placeholders (e.g., 'company_42653').

Sentence: '{sentence}'
Label:"""


prompt2 = """You are a compliance officer specialized in checking sentences for GDPR compliance. 

Your task is to analyze the content of the provided sentence and determine whether it conforms to the Data Protection Officer (DPO) requirement.

Assign a binary score to each sentence:
- 0 if the sentence does not contain relevant DPO-related information.
- 1 if the sentence discloses DPO-relevant information and is compliant to the GDPR requirement.

Consider a sentence compliant (assign label '1') if it either names the Data Protection Officer (DPO) or an 
equivalent authority, or provides their contact details. 

Note: Sentences are also compliant if they refer to some list or enumeration of the required information. 
For example, the sentence 'You can reach our data protection officer via:' is considered compliant, even though the sentence itself does not contain the contact details.

For confidentiality and privacy reasons, the sentences have been anonymized, i.e., numeric values have been randomized,
and names, email addresses, companies and URLs have been substituted with generic placeholders (e.g., 'company_42653').

Example:
Sentence: 'For questions about our privacy and data protection policies, please contact our data protection officer at the given address and contact details.'
Label: 1

Sentence: '{sentence}'
Label:"""


# Define models and prompts to evaluate
models = [
    ("google/flan-t5-xxl", prompt1, "./flan-t5/predictions_prompt1_flan-t5.csv"),
    ("google/flan-t5-xxl", prompt2, "./flan-t5/predictions_prompt2_flan-t5.csv"),
    ("meta-llama/Llama-2-70b-chat-hf", prompt1, "./llama2/predictions_prompt1_llama2.csv"),
    ("meta-llama/Llama-2-70b-chat-hf", prompt2, "./llama2/predictions_prompt2_llama2.csv")
]

# Iterate over each model and prompt
for model_name, prompt_template, output_path in models:
    print(f"Starting evaluation with {model_name}:")
    evaluate_model(data_df, prompt_template, model_name, output_path)
    print(f"Completed evaluation with {model_name}.")
