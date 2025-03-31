from spellchecker import SpellChecker
from nltk.corpus import words
import pandas as pd
import unicodedata
import re
import contractions


class Cleaner:
    def __init__(self):
        self.spell = SpellChecker()
        self.custom_words = {'401k', '401 k', 'k'}
        self.spell.word_frequency.load_words(self.custom_words)
        self.english_words = set(words.words())
        self.english_words = self.english_words.union(self.custom_words)

    # for the benefits column
    def advanced_cleaner(self, text):
        if pd.isnull(text):  # Handle NaN values
            return ""

        # Step 1: Normalize accented letters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        # Step 2: Remove special characters but keep numbers
        text = re.sub(r'[^\w\s]', '', text)

        # Step 3: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 4: Spell-check and keep valid English words or numbers
        filtered_words = []
        for word in text.split():
            if word.isdigit():  # Keep numbers
                filtered_words.append(word)
            elif word in self.english_words:  # Valid English word
                filtered_words.append(word)
            else:  # Attempt spell-check correction
                corrected_word = self.spell.correction(word)
                if corrected_word in self.english_words:  # Check if correction is valid
                    filtered_words.append(corrected_word)

        # Join the filtered words back into a string
        return ' '.join(filtered_words)

    # for the combined profile and description column
    def quick_cleaner(self, text):
        if pd.isnull(text):  # Handle NaN values
            return ""

        # Step 1: Expand contractions
        text = contractions.fix(text)

        # Step 2: Remove special characters (keep letters, numbers, and whitespace)
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def remove_repeats(text):
        if pd.isnull(text):
            return text

        # Remove duplicate words while maintaining order
        words = text.split()
        return ' '.join(dict.fromkeys(words))




# # Parallelized cleaning and spellchecking with tqdm progress bar
# df['cleaned_benefits'] = Parallel(n_jobs=-1)(delayed(clean_and_filter_english)(row) for row in tqdm(df['benefits']), )
# df.drop(columns=['company_profile_and_description'], inplace=True)




