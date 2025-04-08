import pandas as pd
import os
from source.paths import datasets_path,features_path


def load_dataframe(input_path: str):
    """Simply loads the dataset given the filename and the path Loads the dataset"""
    df = pd.read_csv(os.path.join(input_path))

    return df


def save_dataframe(df: pd.DataFrame, file_name: str, is_feature=False):
    """Saves the dataset"""
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


