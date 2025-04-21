import pandas as pd
import unicodedata
import re
import contractions


class Cleaner:
    def __init__(self, df):
        self.df = df
        self.max_word_length = 20  # words with characters more than this are not kept

    def advanced_cleaner(self, text):
        if pd.isnull(text):
            return ""

        # Expand contractions like "don't" to "do not"
        text = contractions.fix(text)

        # Normalize unicode characters (like é → e)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        # Replace special characters with space using vectorized string operations
        text = re.sub(r'[^\w\s]', ' ', text)

        # Split text into words and filter out words that are too long
        words = text.split()
        filtered_words = [word for word in words if len(word) <= self.max_word_length]

        # Join the filtered words back into a single string
        text = ' '.join(filtered_words)

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def perform_advanced_cleaning(self, columns_to_clean):
        for column in columns_to_clean:
            print(f"Cleaning '{column}' column...")

            self.df[f'cleaned_{column}'] = self.df[column].apply(self.advanced_cleaner)

            print(f"Finished cleaning '{column}' column.\n")
            self.df.drop(columns=column, inplace=True)

        return self.df
