from spellchecker import SpellChecker
from nltk.corpus import words
import pandas as pd
import unicodedata
import re
import contractions
from tqdm import tqdm
from joblib import Parallel, delayed
from source.helpers import words_to_skip


# Clean profile and description column
def simple_cleaner(self, text):
    if pd.isnull(text):
        return ""

    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text


class Cleaner:
    def __init__(self, df):
        self.spell = SpellChecker()
        self.custom_words = words_to_skip
        self.spell.word_frequency.load_words(self.custom_words)
        self.english_words = set(words.words()).union(self.custom_words)
        self.df = df

    # Clean benefits column
    def advanced_cleaner(self, text):
        if pd.isnull(text):
            return ""

        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

        filtered_words = []
        for word in text.split():
            if word.isdigit():
                filtered_words.append(word)
            elif word in self.english_words:
                filtered_words.append(word)
            else:
                corrected_word = self.spell.correction(word)
                if corrected_word in self.english_words:
                    filtered_words.append(corrected_word)

        return ' '.join(filtered_words)

    def perform_advanced_cleaning(self):
        print("Cleaning 'benefits' column...")
        self.df['cleaned_benefits'] = Parallel(n_jobs=-1)(
            delayed(self.advanced_cleaner)(text) for text in tqdm(self.df['benefits'])
        )
        return self.df

    # Clean ONLY the profile_and_description column
    def perform_simple_cleaning(self):
        print("Cleaning 'profile_and_description' column...")
        self.df['cleaned_profile_and_description'] = Parallel(n_jobs=-1)(
            delayed(simple_cleaner)(text) for text in tqdm(self.df['profile_and_description'])
        )
        return self.df
