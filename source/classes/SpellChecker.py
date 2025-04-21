# I ended up not using spell correction in the end. This could come in handy if needed at some point

from spellchecker import SpellChecker
from nltk.corpus import words
import pandas as pd
from source.helpers import words_to_skip


class SpellCheckerProcessor:
    def __init__(self):
        """
        Initializes the spell checker processor with English words and custom words.
        """
        self.custom_words = words_to_skip
        self.english_words = set(words.words()).union(self.custom_words)
        self.spell = SpellChecker()
        self.spell.word_frequency.load_words(self.custom_words)
        self.correction_cache = {}

    def apply_spellcheck(self, word_list):
        """
        Takes a list of words and returns a corrected list (only for words not in English dictionary).

        :param word_list: List of words to spellcheck.
        :return: List of corrected words.
        """
        corrected_words = []
        for word in word_list:
            if word.isdigit() or word in self.english_words:
                corrected_words.append(word)
            else:
                if word in self.correction_cache:
                    corrected_word = self.correction_cache[word]
                else:
                    corrected_word = self.spell.correction(word)
                    self.correction_cache[word] = corrected_word

                if corrected_word in self.english_words:
                    corrected_words.append(corrected_word)
        return corrected_words

    def perform_spellcheck(self, df, columns_to_check):
        """
        Perform spellchecking on the specified columns and return the dataframe with corrected columns.

        :param df: DataFrame containing the text columns to spellcheck.
        :param columns_to_check: List of column names to perform spellchecking on.
        :return: DataFrame with corrected text in new columns.
        """
        for column in columns_to_check:
            # Apply spellcheck to each row in the specified column
            df[f'corrected_{column}'] = df[column].apply(
                lambda x: " ".join(self.apply_spellcheck(str(x).split())) if pd.notnull(x) else "")

        return df
