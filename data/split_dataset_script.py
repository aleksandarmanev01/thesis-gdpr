import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('GDPR.csv', sep='\t')

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['DPO'])

# Save the training data to a new CSV file
train_data.to_csv('training_data_dpo.csv', index=False)

print("Training data has been saved to 'training_data_dpo.csv'")
