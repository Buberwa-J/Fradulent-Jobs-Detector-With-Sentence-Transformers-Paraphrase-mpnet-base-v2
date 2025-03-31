import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
import os


semi_cleaned_df = pd.read_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\fake_job_postings.csv")
uncleaned_df = pd.read_csv(r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\emscad_v1.csv")


df = semi_cleaned_df.drop(columns=['job_id',
                                   'location',
                                   'department',
                                   'telecommuting',
                                   'has_company_logo',
                                   'has_questions'])

# 1. Combine the 'title' and the 'function' columns.
# 2. Rename the 'industry' column to a more descriptive name.
# 3. Lowercase everything to make mapping easier from now on


df['nature_of_job'] = df['title'] + ' ' + df['function']
df.drop(columns=['title', 'function'], inplace=True)

df.rename(columns={'industry': 'nature_of_company',
                   'required_experience': 'type_of_position',
                   'employment_type': 'type_of_contract'
                   }, inplace=True)

# Apply .str.lower() to all string columns in the DataFrame
df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').apply(lambda x: x.str.lower())

df.to_csv(r'../datasets/preliminary_cleaned_df.csv', index=False)
