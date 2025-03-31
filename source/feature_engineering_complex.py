import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from spellchecker import SpellChecker
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import contractions
from configurations import embedding_model_path

# Load the dataframe from the checkpoint
df = pd.read_csv(
    r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_one.csv")


# # Embedd the following textual columns
# 1. Company profile and description
# 2. Requirements
# 3. Nature of company
# 4. Nature of job
# 5. type of position

# In[39]:


model_path = embedding_model_path

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