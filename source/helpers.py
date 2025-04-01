import pandas as pd
import os


def load_dataframe(input_path: str):
    """Simply loads the dataset given the filename and the path Loads the dataset"""
    df = pd.read_csv(os.path.join(input_path))

    return df


def save_dataframe(df: pd.DataFrame, output_path: str, file_name: str):
    """Saves the cleaned dataset"""
    output_file = os.path.join(output_path, file_name)
    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to {output_file}")


