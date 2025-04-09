import pandas as pd
from source.helpers import load_dataframe


class FeatureSelector:
    def __init__(self):
        self.df = load_dataframe('fully_cleaned_dataframe')
        self.y = self.df['fraudulent']

    def make_df_from_features(self, feature_space):
        dfs = []

        for feature in feature_space:
            feature_df = load_dataframe(feature, is_feature=True)
            dfs.append(feature_df)

        # Concatenate all features horizontally
        combined_df = pd.concat(dfs, axis=1)

        # Add the target variable
        combined_df['fraudulent'] = self.y.reset_index(drop=True)

        return combined_df
