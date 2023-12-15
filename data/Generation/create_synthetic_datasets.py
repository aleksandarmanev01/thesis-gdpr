import pandas as pd


def read_txt_to_df(txt_file):
    # Read the text file and create a DataFrame
    with open(txt_file, 'r') as file:
        sentences = file.readlines()
    df = pd.DataFrame(sentences, columns=['Text'])
    df['Text'] = df['Text'].str.strip()  # Remove any leading/trailing whitespace
    return df


def create_and_save_subsets(df, sizes, base_file_name):
    for size in sizes:
        subset = df.head(size).copy()
        subset['DPO'] = 1  # Add the 'DPO' column with all values set to 1
        file_name = f"{base_file_name}_{size}.csv"
        subset.to_csv(file_name, index=False)
        print(f"Saved {file_name} with {size} records")


# Sizes for the datasets
sizes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
base_file_name = 'generated_dpo'

# Read data from text file
df = read_txt_to_df('dpo_compliant.txt')

# Remove duplicates and shuffle with a random seed for reproducibility
df = df.drop_duplicates()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Create and save subsets
create_and_save_subsets(df, sizes, base_file_name)
