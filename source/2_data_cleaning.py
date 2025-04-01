import pandas as pd

# Clean the benefits column before engineering anything else
# 1. Convert accented letters into normal english letters
# 2. Remove special characters but keep the '401k' since I think it's critical to have
# 3. Since the extraction of the benefits was very hectic I think I should spell check and keep only the correct words
# 4. It is very important I run this once and store the results somewhere else since it is resource intensive and re-running it would be a waste of time


df = pd.read_csv(r'../datasets/preliminary_cleaned_df.csv')


