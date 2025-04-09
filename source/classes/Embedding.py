import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from source.helpers import save_dataframe
from source.paths import embedding_model_path


class Embedding:
    def __init__(self, df, n_components=300):
        """
        Initializes the embedding class with a dataframe, embedding model path, and PCA component count.

        :param df: The dataframe containing the text columns.
        :param n_components: The number of PCA components to keep.
        """
        self.model_path = embedding_model_path
        self.model = SentenceTransformer(self.model_path)
        self.df = df
        self.n_components = n_components  # Number of components for PCA

    # Function to create embeddings for the specified columns and apply PCA
    def embed(self, columns):
        for column in columns:
            print(f"Creating embeddings for '{column}' column...")

            # Generate embeddings for the column
            embeddings = self.df[column].apply(self._embed_text)

            # Convert embeddings to a DataFrame
            embeddings_df = pd.DataFrame(embeddings.tolist())

            # Apply PCA to reduce the dimensionality of the embeddings
            pca = PCA(n_components=self.n_components)
            reduced_embeddings = pca.fit_transform(embeddings_df)

            # Create a DataFrame for the reduced embeddings
            reduced_embeddings_df = pd.DataFrame(reduced_embeddings, columns=[f"{column}_pca_{i}" for i in range(self.n_components)])

            # Convert the reduced embeddings to float32 to save memory
            reduced_embeddings_df = reduced_embeddings_df.astype('float32')

            # Save the reduced embeddings using the provided save_dataframe method
            embeddings_filename = f"{column}_embeddings_reduced.csv"
            save_dataframe(reduced_embeddings_df, embeddings_filename, is_feature=True)

    # Helper method to create embeddings for a single text
    def _embed_text(self, text):
        if not text or pd.isnull(text):
            return np.zeros(self.model.get_sentence_embedding_dimension())
        text = str(text).lower()
        return self.model.encode(text)

