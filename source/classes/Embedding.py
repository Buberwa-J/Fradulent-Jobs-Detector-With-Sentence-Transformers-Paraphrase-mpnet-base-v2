import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from source.helpers import save_dataframe
from source.paths import embedding_model_path


class Embedding:
    def __init__(self, df, n_components=None):
        """
        Initializes the embedding class with a dataframe and PCA component count.

        :param df: The dataframe containing the text columns.
        :param n_components: The number of PCA components to keep. If None, PCA will not be applied.
        """
        self.model_path = embedding_model_path
        self.model = SentenceTransformer(self.model_path)
        self.df = df
        self.n_components = n_components

    def embed(self, columns):
        for column in columns:
            print(f"Creating embeddings for '{column}' column...")

            embeddings = self.df[column].apply(self._embed_text)
            embeddings_df = pd.DataFrame(embeddings.tolist())

            if self.n_components and self.n_components > 0:
                pca = PCA(n_components=self.n_components)
                reduced_embeddings = pca.fit_transform(embeddings_df)
                reduced_embeddings_df = pd.DataFrame(
                    reduced_embeddings,
                    columns=[f"{column}_pca_{i}" for i in range(self.n_components)]
                )
                reduced_embeddings_df = reduced_embeddings_df.astype('float32')
                embeddings_filename = f"{column}_embeddings_reduced.csv"
                save_dataframe(reduced_embeddings_df, embeddings_filename, is_feature=True)
            else:
                embeddings_df = embeddings_df.astype('float32')
                embeddings_filename = f"{column}_embeddings.csv"
                save_dataframe(embeddings_df, embeddings_filename, is_feature=True)

    def embed_for_inference(self, columns):
        """
        Stateless method to generate embeddings for inference across multiple columns.
        Each embedded column will have the same column names: 0 to (embedding_dim - 1),
        just like the saved CSVs during training.

        :param columns: List of column names in the dataframe to embed.
        :return: DataFrame of concatenated embeddings with repeated integer column names.
        """
        all_embeddings = []

        for column in columns:
            print(f"Generating embeddings for inference on '{column}'...")

            texts = self.df[column].fillna("").astype(str).str.lower().tolist()
            embeddings = self.model.encode(texts)

            if self.n_components and self.n_components > 0:
                pca = PCA(n_components=self.n_components)
                embeddings = pca.fit_transform(embeddings)
                embedding_dim = self.n_components
            else:
                embedding_dim = self.model.get_sentence_embedding_dimension()

            col_embedding_df = pd.DataFrame(
                embeddings,
                columns=list(range(embedding_dim))
            )

            all_embeddings.append(col_embedding_df)

        final_embeddings_df = pd.concat(all_embeddings, axis=1)
        return final_embeddings_df.astype('float32')

    def _embed_text(self, text):
        if not text or pd.isnull(text):
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(str(text).lower())
