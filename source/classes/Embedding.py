import  pandas as pd
from source.paths import embedding_model_path
from sentence_transformers import SentenceTransformer
import numpy as np
import re


class Embedding:
    def __init__(self):
        self.model_path = embedding_model_path
        self.model = SentenceTransformer(self.model_path)

    # Function to clean text and create embeddings
    def clean_and_embed(self, text):
        if not text or pd.isnull(text):
            return np.zeros(self.model.get_sentence_embedding_dimension())
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return self.model.encode(text)





# # Embedd the following textual columns
# 1. Company profile and description
# 2. Requirements
# 3. Nature of company
# 4. Nature of job
# 5. type of position

   # Example of the function call
# # Clean and create embeddings for 'benefits'
# df['benefits_embeddings'] = df['benefits'].progress_apply(clean_and_embed)
# benefits_embeddings_df = pd.DataFrame(df['benefits_embeddings'].tolist()).astype('float32')
# benefits_embeddings_df.to_csv(os.path.join(output_dir, 'benefits_embeddings.csv'), index=False)