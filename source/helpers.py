import pandas as pd
import os
from source.paths import datasets_path,features_path


def load_dataframe(dataframe_name: str, is_feature=False):
    if is_feature:

        if dataframe_name == 'salary':
            dataframe_name = 'salary_features_dataframe.csv'
        else:
            dataframe_name = 'cleaned_' + dataframe_name + '_embeddings_reduced.csv'

        df = pd.read_csv(os.path.join(features_path, dataframe_name))
    else:
        dataframe_name = dataframe_name + '.csv'
        df = pd.read_csv(os.path.join(datasets_path, dataframe_name))

    return df


def save_dataframe(df: pd.DataFrame, file_name: str, is_feature=False):
    """Saves the dataset"""
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    if is_feature:
        output_file = os.path.join(features_path, file_name)
    else:
        output_file = os.path.join(datasets_path, file_name)
    df.to_csv(output_file, index=False)
    print(f"{file_name} saved successfully")


def remove_repeats(text):
    if pd.isnull(text):
        return text

    # Remove duplicate words while maintaining order
    words = text.split()
    return ' '.join(dict.fromkeys(words))


words_to_skip = {'401k', '401 k', 'k'}

do_embedding = False


