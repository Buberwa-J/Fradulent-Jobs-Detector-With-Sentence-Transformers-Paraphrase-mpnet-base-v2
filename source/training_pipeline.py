# training_pipeline.py
from source.classes.PreliminaryCleaner import PreliminaryCleaner
from source.paths import original_dataframe_path
from source.helpers import load_dataframe, save_dataframe, remove_repeats
from source.classes.SalaryFeatureExtractor import *


def main():
    original_dataframe = load_dataframe(original_dataframe_path)

    print("Starting the Job Legitimacy Detector Pipeline...\n")

    # Step 1: Preliminary Cleaning
    print("Doing some preliminary cleaning on the data...")
    cleaner = PreliminaryCleaner(original_dataframe)
    preliminary_cleaned_dataframe = cleaner.clean_data()

    # The contents of nature_of_job are very repetitive, keep only unique occurrences
    preliminary_cleaned_dataframe['nature_of_job'] = preliminary_cleaned_dataframe['nature_of_job'].apply(remove_repeats)
    save_dataframe(preliminary_cleaned_dataframe,  'preliminary_cleaned_dataframe.csv')
    print("Preliminary cleaning done successfully\n")

    # Step 2 (Optional):
    print("Extracting simple features from the salary")
    salary_feature_extractor = SalaryFeatureExtractor(preliminary_cleaned_dataframe)
    salary_feature_extractor.process()
    simple_features_dataframe = preliminary_cleaned_dataframe
    save_dataframe(simple_features_dataframe, 'simple_feature_dataframe.csv')
    print("Salary features extracted successfully\n")


if __name__ == "__main__":
    main()
