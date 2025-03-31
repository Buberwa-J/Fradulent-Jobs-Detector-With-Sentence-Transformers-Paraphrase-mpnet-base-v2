# Clean the benefits column before engineering anything else
# 1. Convert accented letters into normal english letters
# 2. Remove special characters but keep the '401k' since I think it's critical to have
# 3. Since the extraction of the benefits was very hectic I think I should spell check and keep only the correct words
# 4. It is very important I run this once and store the results somewhere else since it is resource intensive and re-running it would be a waste of time

# Initialize SpellChecker and add custom words
spell = SpellChecker()
custom_words = {'401k', '401 k', 'k'}
spell.word_frequency.load_words(custom_words)

# List of valid English words
english_words = set(words.words())
english_words = english_words.union(custom_words)


# Define a function to clean the text
# Cleaning and Spellchecking Function
def clean_and_filter_english(text):
    if pd.isnull(text):  # Handle NaN values
        return ""

    # Step 1: Normalize accented letters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Step 2: Remove special characters but keep numbers
    text = re.sub(r'[^\w\s]', '', text)

    # Step 3: Convert to lowercase
    text = text.lower()

    # Step 4: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 5: Spell-check and keep valid English words or numbers
    filtered_words = []
    for word in text.split():
        if word.isdigit():  # Keep numbers
            filtered_words.append(word)
        elif word in english_words:  # Valid English word
            filtered_words.append(word)
        else:  # Attempt spell-check correction
            corrected_word = spell.correction(word)
            if corrected_word in english_words:  # Check if correction is valid
                filtered_words.append(corrected_word)

    # Join the filtered words back into a string
    return ' '.join(filtered_words)

# Parallelized cleaning and spellchecking with tqdm progress bar
df['cleaned_benefits'] = Parallel(n_jobs=-1)(delayed(clean_and_filter_english)(row) for row in tqdm(df['benefits']), )
df.drop(columns=['company_profile_and_description'], inplace=True)

def clean_company_profile_and_description(text):
    if pd.isnull(text):  # Handle NaN values
        return ""

    # Step 1: Expand contractions
    text = contractions.fix(text)

    # Step 2: Remove special characters (keep letters, numbers, and whitespace)
    text = re.sub(r'[^\w\s]', '', text)

    return text


def remove_repeats(text):
    # Check if the input is NaN, if so return it as is
    if pd.isna(text):
        return text

    # Create a list of the words in the column
    words = text.split()

    # Create a dictionary with unique keys only
    unique_word = list(dict.fromkeys(words))

    # Join the words back
    return ' '.join(unique_word)