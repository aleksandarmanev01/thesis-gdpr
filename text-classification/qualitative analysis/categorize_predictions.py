import pandas as pd

# File names
files = [
    "../flan-t5/predictions_prompt1_flan-t5.csv",
    "../flan-t5/predictions_prompt2_flan-t5.csv",
    "../llama2/predictions_prompt1_llama2.csv",
    "../llama2/predictions_prompt2_llama2.csv"
]

# Read the datasets
datasets = [pd.read_csv(f) for f in files]


# Function to categorize data into TP, FP, FN, TN
def categorize_data(row):
    if row['actual_label'] == 1:
        return 'TP' if row['predicted_label'] == 1 else 'FN'
    else:
        return 'TN' if row['predicted_label'] == 0 else 'FP'


# Apply the categorization function to all datasets
for dataset in datasets:
    dataset['category'] = dataset.apply(categorize_data, axis=1)

# Merge the datasets
merged = pd.concat(datasets)

# Group by sentence, predicted_label, actual_label, and category, and count occurrences
grouped = merged.groupby(['sentence', 'predicted_label', 'actual_label', 'category']).size().reset_index(name='counts')

# Filter to keep only those sentences where the category is consistent across all datasets
consistent = grouped[grouped['counts'] == len(files)]

# Split into individual categories
fp = consistent[consistent['category'] == 'FP'][['sentence', 'predicted_label', 'actual_label']]
fn = consistent[consistent['category'] == 'FN'][['sentence', 'predicted_label', 'actual_label']]
tp = consistent[consistent['category'] == 'TP'][['sentence', 'predicted_label', 'actual_label']]
tn = consistent[consistent['category'] == 'TN'][['sentence', 'predicted_label', 'actual_label']]

# Save the results
fp.to_csv('FP.csv', index=False)
fn.to_csv('FN.csv', index=False)
tp.to_csv('TP.csv', index=False)
tn.to_csv('TN.csv', index=False)
