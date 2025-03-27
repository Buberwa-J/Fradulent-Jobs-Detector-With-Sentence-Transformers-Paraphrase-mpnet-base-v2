#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
from nltk.corpus import words
import textdistance
from tqdm import tqdm
from spellchecker import SpellChecker
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import contractions
import os

# # Import the dataset which is locally available

# In[160]:


semi_cleaned_df = pd.read_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\fake_job_postings.csv")
uncleaned_df = pd.read_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\emscad_v1.csv")

# # Get a glimpse of what it looks like and pick columns that seem useful

# In[162]:


semi_cleaned_df.head()

# # These columns don't seem useful at all. Drop them. Also fill missing values with np.nan
#

# In[166]:


df = semi_cleaned_df.drop(columns=['job_id',
                                   'location',
                                   'department',
                                   'telecommuting',
                                   'has_company_logo',
                                   'has_questions'])

# # Make these minor changes to the columns
# 1. Combine the 'title' and the 'function' columns.
# 2. Rename the 'industry' column to a more descriptive name.
# 3. Lowercase everything to make mapping easier from now on

# In[169]:


df['nature_of_job'] = df['title'] + ' ' + df['function']
df.drop(columns=['title', 'function'], inplace=True)

df.rename(columns={'industry': 'nature_of_company',
                   'required_experience': 'type_of_position',
                   'employment_type': 'type_of_contract'
                   }, inplace=True)

# Apply .str.lower() to all string columns in the DataFrame
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.str.lower())

# # See how the dataset looks with these polished columns

# In[172]:


df.head()


# # The 'nature_of_job'  and the 'nature_of_company' need some cleaning
# 1. The contents of the 'nature_of_job' contain repetitive words. Keep only unique occurences

# In[175]:


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


# Apply the function to the 'nature_of_job' column
df['nature_of_job'] = df['nature_of_job'].apply(remove_repeats)

# # A little cleanup for the 'salary_range' column
# 1. The <u>uncleaned df </u> has the salary_range column that is cleaner than that of the <u> semi cleaned df </u>
# 2. Extract the minimum and the maximum salary

# In[178]:


# Step 1: Use the column from the uncleaned dataset
df['salary_range'] = uncleaned_df['salary_range']


# Step 2: Function to extract min and max values and return as floats
def extract_min_max(salary):
    try:
        # Split the range into min and max values
        low, high = map(float, salary.split('-'))
        return f"{low:.1f}", f"{high:.1f}"  # Format as readable floats
    except (ValueError, AttributeError):
        # Handle cases where extraction is impossible "Not Mentioned' also falls under this categ
        return None, None


# Apply the function to extract and format salary_min and salary_max
df[['salary_min', 'salary_max']] = df['salary_range'].apply(lambda x: pd.Series(extract_min_max(x)))


# Step 3: Define a function to determine the type of salary
def determine_salary_type(salary_max):
    try:
        # Convert salary_max to a float if it's not already
        salary = float(salary_max)

        # Flags for different salary types
        # These ranges are very subjective. I could definately do some more digging
        is_hourly = 0 < salary < 500.0
        is_monthly = 500.0 <= salary <= 30000.0
        is_annual = salary > 30000.0

        return is_hourly, is_monthly, is_annual
    except (ValueError, TypeError):
        # Handle cases where salary_max is invalid or missing
        return False, False, False


# Apply the function and unpack the results into new columns
df[['is_hourly', 'is_monthly', 'is_annual']] = df['salary_max'].apply(
    lambda x: pd.Series(determine_salary_type(x))
)

# Ensure the extracted salary columns are of float type (in case they were not properly converted)
df['salary_min'] = df['salary_min'].astype('float32')
df['salary_max'] = df['salary_max'].astype('float32')

# In[180]:


df.required_education.unique()

# # Engineer some features from the 'required_education' column.
# 1. Since I have few unique values, I could just easily create a dictionary with maps. These values are also subjective so I could definately dig abit deeper

# In[183]:


education_bins = {
    'unspecified': np.nan,
    'high school or equivalent': 2,
    'some high school coursework': 2,
    'vocational - hs diploma': 2,
    'vocational': 3,
    'vocational - degree': 3,
    'certification': 3,
    'some college coursework completed': 4,
    'associate degree': 4,
    "bachelor's degree": 5,
    'professional': 5,
    "master's degree": 7,
    "doctorate": 8,
}

df['required_education_numeric'] = df['required_education'].map(education_bins).astype('float16')

# In[185]:


df.type_of_position.unique()

# # Engineer some features from the 'type_of_position' column.
# 1. Since I have few unique values, I could just easily create a dictionary with maps just like in the education. These values are also subjective so I could definately dig abit deeper

# In[188]:


# Define the mapping
position_mapping = {
    'not applicable': np.nan,
    'internship': 2,
    'entry level': 3,
    'associate': 3,
    'mid-senior level': 4,
    'director': 5,
    'executive': 6
}

# Map the values
df['type_of_position_numeric'] = df['type_of_position'].map(position_mapping).astype('float16')

# In[190]:


df.type_of_contract.unique()

# # Create some encoding for the type_of_contract feature. I've chosen one-hot since I don't want to risk creating something ordinal where there shouldn't be anything ordinal

# In[193]:


# Perform one-hot encoding
df = pd.get_dummies(df, columns=['type_of_contract'], drop_first=True)

# # Create new features based on the minimum salary mentioned
# 1. Product of minimum salary and required_education_numeric
# 2. Product of minimum salary and type_of_position_numeric

# In[196]:


# Salary * Education
df['salary_education_min_product'] = (df['salary_min'] * df['required_education_numeric'])
df['salary_education_max_product'] = (df['salary_max'] * df['required_education_numeric'])

# Salary * position
df['salary_position_min_product'] = (df['salary_min'] * df['type_of_position_numeric'])
df['salary_position_max_product'] = (df['salary_max'] * df['type_of_position_numeric'])

# # Clean the benefits column before engineering anything else
# 1. Convert accented letters into normal english letters
# 2. Remove special characters but keep the '401k' since I think it's critical to have
# 3. Since the extraction of the benefits was very hectic I think I should spell check and keep only the correct words
# 4. It is very important I run this once and store the results somewhere else since it is resource intensive and re-running it would be a waste of time

# # Initialize SpellChecker and add custom words
# spell = SpellChecker()
# custom_words = {'401k', '401 k', 'k'}
# spell.word_frequency.load_words(custom_words)
#
# # List of valid English words
# english_words = set(words.words())
# english_words = english_words.union(custom_words)
#
# # Cleaning and Spellchecking Function
# def clean_and_filter_english(text):
#     if pd.isnull(text):  # Handle NaN values
#         return ""
#
#     # Step 1: Normalize accented letters
#     text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
#
#     # Step 2: Remove special characters but keep numbers
#     text = re.sub(r'[^\w\s]', '', text)
#
#     # Step 3: Convert to lowercase
#     text = text.lower()
#
#     # Step 4: Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     # Step 5: Spell-check and keep valid English words or numbers
#     filtered_words = []
#     for word in text.split():
#         if word.isdigit():  # Keep numbers
#             filtered_words.append(word)
#         elif word in english_words:  # Valid English word
#             filtered_words.append(word)
#         else:  # Attempt spell-check correction
#             corrected_word = spell.correction(word)
#             if corrected_word in english_words:  # Check if correction is valid
#                 filtered_words.append(corrected_word)
#
#     # Join the filtered words back into a string
#     return ' '.join(filtered_words)
#
# # Parallelized cleaning and spellchecking with tqdm progress bar
# tqdm.pandas()  # Enable progress_apply
# df['cleaned_benefits'] = Parallel(n_jobs=-1)(delayed(clean_and_filter_english)(row) for row in tqdm(df['benefits']), )
# df.drop(columns=['company_profile_and_description'], inplace=True)

# # I already ran the above code once and then i stored the output in a different dataframe. I should load it before I save the new dataframe as a checkpoint

# In[201]:


# Step 1: Load the cleaned_benefits.csv
cleaned_benefits_df = pd.read_csv(
    "D:\\Machine Learning Approach To Job Legitimacy Detector\\Data\\Pre-features\\cleaned_benefits.csv")

# Step 2: Drop the index column (assuming it's the unnamed column that was saved as part of the CSV)
# If the index column is the first column (unnamed), drop it
cleaned_benefits_df = cleaned_benefits_df.drop(cleaned_benefits_df.columns[0], axis=1)

# Step 3: Concatenate the cleaned_benefits with the original df
df = pd.concat([df, cleaned_benefits_df], axis=1)

# # Handle the company_profile and description columns
# 1. Join the two columns to create a new column
# 2. Fix contractions
# 3. Remove special characters but keep the numbers and every other thing

# In[204]:


df['company_profile_and_description'] = df['company_profile'] + ' ' + df['description']
df.drop(columns=['description', 'company_profile'], inplace=True)

# Enable tqdm for pandas progress_apply
tqdm.pandas()


# Define a function to clean the text
def clean_company_profile_and_description(text):
    if pd.isnull(text):  # Handle NaN values
        return ""

    # Step 1: Expand contractions
    text = contractions.fix(text)

    # Step 2: Remove special characters (keep letters, numbers, and whitespace)
    text = re.sub(r'[^\w\s]', '', text)

    return text


# Apply the cleaning function to the column
df['cleaned_company_profile_and_description'] = df['company_profile_and_description'].progress_apply(
    clean_company_profile_and_description)

df.drop(columns=['company_profile_and_description'], inplace=True)

# # Save the dataframe from the checkpoint

# In[207]:


df.to_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_one.csv",
          index=False)

# ## Load the dataframe from the checkpoint

# In[210]:


# Load the dataframe from the checkpoint
df = pd.read_csv(
    r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_one.csv")
df

# # Embedd the following textual columns
# 1. Company profile and description
# 2. Requirements
# 3. Nature of company
# 4. Nature of job
# 5. type of position

# In[39]:


model_path = r'C:\Users\hp\models\paraphrase-mpnet-base-v2\models--sentence-transformers--paraphrase-mpnet-base-v2\snapshots\bef3689366be4ad4b62c8e1cec013639bea3c86a'

# Load the model from the local directory
model = SentenceTransformer(model_path)

# Initialize tqdm for progress tracking
tqdm.pandas()


# Function to clean text and create embeddings
def clean_and_embed(text):
    if not text or pd.isnull(text):
        return np.zeros(model.get_sentence_embedding_dimension())
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return model.encode(text)


#  the output directory
output_dir = "D:\Machine Learning Approach To Job Legitimacy Detector\Data\Embedded Features"

# Clean and create embeddings for 'type_of_position'
df['type_of_position_embeddings'] = df['type_of_position'].progress_apply(clean_and_embed)
type_of_position_embeddings_df = pd.DataFrame(df['type_of_position_embeddings'].tolist()).astype('float32')
type_of_position_embeddings_df.to_csv(os.path.join(output_dir, 'type_of_position_embeddings.csv'), index=False)

# Clean and create embeddings for 'requirements'
df['requirements_embeddings'] = df['requirements'].progress_apply(clean_and_embed)
requirements_embeddings_df = pd.DataFrame(df['requirements_embeddings'].tolist()).astype('float32')
requirements_embeddings_df.to_csv(os.path.join(output_dir, 'requirements_embeddings.csv'), index=False)

# Clean and create embeddings for 'nature_of_company'
df['nature_of_company_embeddings'] = df['nature_of_company'].progress_apply(clean_and_embed)
nature_of_company_embeddings_df = pd.DataFrame(df['nature_of_company_embeddings'].tolist()).astype('float32')
nature_of_company_embeddings_df.to_csv(os.path.join(output_dir, 'nature_of_company_embeddings.csv'), index=False)

# Clean and create embeddings for 'nature_of_job'
df['nature_of_job_embeddings'] = df['nature_of_job'].progress_apply(clean_and_embed)
nature_of_job_embeddings_df = pd.DataFrame(df['nature_of_job_embeddings'].tolist()).astype('float32')
nature_of_job_embeddings_df.to_csv(os.path.join(output_dir, 'nature_of_job_embeddings.csv'), index=False)

# Clean and create embeddings for 'benefits'
df['benefits_embeddings'] = df['benefits'].progress_apply(clean_and_embed)
benefits_embeddings_df = pd.DataFrame(df['benefits_embeddings'].tolist()).astype('float32')
benefits_embeddings_df.to_csv(os.path.join(output_dir, 'benefits_embeddings.csv'), index=False)

# Clean and create embeddings for 'company_profile_and_description'
df['company_profile_and_description_embeddings'] = df['cleaned_company_profile_and_description'].progress_apply(
    clean_and_embed)
company_profile_embeddings_df = pd.DataFrame(df['company_profile_and_description_embeddings'].tolist()).astype(
    'float32')
company_profile_embeddings_df.to_csv(os.path.join(output_dir, 'company_profile_and_description_embeddings.csv'),
                                     index=False)

# I won't need these columns since they are already stored in different files
df.drop(columns=[
    'type_of_position',
    'requirements',
    'nature_of_company',
    'nature_of_job',
    'benefits',
    'cleaned_company_profile_and_description'
], inplace=True)

# In[ ]:


# In[ ]:


# In[215]:


print(df.dtypes)

# # Only keep the columns that are <b>NOT</b> found in separate files. This includes removing all embeddings columns It also involves removing columns that I wont use as features

# In[18]:


df.drop(columns=[
    'salary_range',
    'requirements',
    'type_of_position',
    'required_education',
    'nature_of_company',
    'nature_of_job',
    'benefits',
    'cleaned_benefits',
    'cleaned_company_profile_and_description'
], inplace=True)

# # Create another checkpoint with the dataframe

# In[221]:


df.to_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_two.csv",
          index=False)

# # Load the data from the second checkpoint

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import unicodedata
from nltk.corpus import words
import textdistance
from tqdm import tqdm
from spellchecker import SpellChecker
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import contractions
import os

df = pd.read_csv(
    r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_two.csv")
df

# In[5]:


# Directory containing embeddings
embedding_dir = r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Embedded Features"

# List of embedding files
embedding_files = [
    "benefits_embeddings.csv",
    "company_profile_and_description_embeddings.csv",
    "nature_of_company_embeddings.csv",
    "nature_of_job_embeddings.csv",
    "requirements_embeddings.csv",
    "type_of_position_embeddings.csv"
]

# Dictionary to store renamed DataFrames
renamed_embeddings = {}

# Process each embedding file
for file in embedding_files:
    file_path = os.path.join(embedding_dir, file)

    # Read CSV
    df_embedding = pd.read_csv(file_path)

    # Extract feature name from filename
    feature_name = file.replace("_embeddings.csv", "")

    # Rename columns systematically
    df_embedding.columns = [f"{feature_name}_dim_{i}" for i in range(df_embedding.shape[1])]

    # Store in dictionary
    renamed_embeddings[feature_name] = df_embedding

# Concatenate all embeddings horizontally
df_embeddings = pd.concat(renamed_embeddings.values(), axis=1)

# Merge with the main dataset
df = pd.concat([df, df_embeddings], axis=1)

# Check the final dataset
df

# In[ ]:


# In[ ]:


# # Without perfoming any oversampling and using balanced class weights

# In[280]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

# 1. Load dataset (assuming 'target' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train HistGradientBoostingClassifier (handles NaNs automatically)
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
hgb.fit(X_train, y_train)

# 4. Make predictions
y_pred = hgb.predict(X_test)

# 5. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Using balanced class weights

# In[284]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Train HistGradientBoostingClassifier with class_weight='balanced'
hgb = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
hgb.fit(X_train_resampled, y_train_resampled)

# 6. Make predictions
y_pred = hgb.predict(X_test)

# 7. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Oversampling and using balanced class weights

# In[286]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the target variable)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Train RandomForestClassifier with class_weight='balanced' to handle class imbalance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_resampled, y_train_resampled)

# 6. Make predictions
y_pred = rf.predict(X_test)

# 7. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:


# # Using calculated class weights and oversampling

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# 1. Load dataset (assuming 'fraudulent' is the column to predict)
# df = pd.read_csv('your_dataset.csv')

# Example: Simulated dataset
X = df.drop(columns='fraudulent')  # Features
y = df['fraudulent']  # Target variable

# 2. Replace NaN values with 0
X.fillna(0, inplace=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Apply SMOTE to oversample the minority class in the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Compute manual class weights (giving more weight to the minority class)
class_weights = {0: 1., 1: len(y_train) / (2 * np.sum(y_train == 1))}

# 6. Train HistGradientBoostingClassifier with manually calculated class weights
hgb = HistGradientBoostingClassifier(class_weight=class_weights, random_state=42)
hgb.fit(X_train_resampled, y_train_resampled)

# 7. Make predictions
y_pred = hgb.predict(X_test)

# 8. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# In[ ]:




