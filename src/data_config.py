import pandas as pd

DATASET_PATH = '../data/GDPR.csv'
AUGMENTED_FILES = [f'../data/Augmentation/augmented_dpo_compliant_{i}.csv' for i in range(1, 6)]
GENERATED_FILES = [f'../data/Generation/generated_dpo_{i * 300}.csv' for i in range(1, 11)]
LSTM_LOG_FILE_PATH = './lstm/training_data/training_logs/'

scenarios = {
    'Scenario_OAC_O': AUGMENTED_FILES,
    'Scenario_OGC_O': GENERATED_FILES
}


def config_scenario(generated_data, original_train, original_test, scenario_type):
    """
       Configure the training and test datasets based on the specified scenario type.

       Parameters:
       - generated_data (DataFrame): The dataframe containing the generated or augmented data.
       - original_train (DataFrame): The dataframe containing the original training data.
       - original_test (DataFrame): The dataframe containing the original test data.
       - scenario_type (str): The type of scenario to configure datasets for.

       Returns:
       - train (DataFrame): Configured training data for the scenario.
       - test (DataFrame): Configured test data for the scenario.
       """
    if scenario_type == "Scenario_OAC_O" or scenario_type == "Scenario_OGC_O":
        train = pd.concat([original_train, generated_data], axis=0).sample(frac=1).reset_index(drop=True)
        test = original_test
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return train, test


def preprocess_data(data_set, preprocessing_func):
    """
        Preprocess the dataset and split it into features and labels.

        Parameters:
        - data_set (DataFrame): The data to preprocess.
        - preprocessing_func (function): The function to apply on the 'message' column.

        Returns:
        - X (Series): The processed 'message' column.
        - y (Series): The 'label' column.
    """
    data_set['Text'] = data_set['Text'].apply(preprocessing_func)
    data_set['DPO'] = data_set['DPO']

    X, y = data_set['Text'], data_set['DPO']
    X, y = drop_empty_rows_and_reset_index(X, y)
    return X, y


def drop_empty_rows_and_reset_index(df_X, df_y):
    empty_rows = df_X.index[df_X == '']
    if not empty_rows.empty:
        df_X.drop(empty_rows, inplace=True)
        df_y.drop(empty_rows, inplace=True)
    return df_X.reset_index(drop=True), df_y.reset_index(drop=True)
