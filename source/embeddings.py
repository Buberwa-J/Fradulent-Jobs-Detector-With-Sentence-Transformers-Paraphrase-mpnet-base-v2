import pandas as pd

import os

df = pd.read_csv(
    r"D:\Machine Learning Approach To Job Legitimacy Detector\Data\Checkpoint Dataframes\checkpoint_two.csv")


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

