# training_pipeline.py
from source.classes.PreliminaryCleaner import PreliminaryCleaner
from source.classes.Embedding import Embedding
from source.classes.Cleaner import Cleaner
from source.helpers import load_dataframe, save_dataframe, remove_repeats, do_embedding
from source.classes.SalaryFeatureExtractor import *


def main():
    original_dataframe = load_dataframe('emscad_without_tags')

    print("Starting the Job Legitimacy Detector Pipeline...\n")

    # Step 1: Preliminary Cleaning
    print("Doing some preliminary cleaning on the data...")
    preliminary_cleaner = PreliminaryCleaner(original_dataframe)
    preliminary_cleaned_dataframe = preliminary_cleaner.clean_data()

    # The contents of nature_of_job are very repetitive, keep only unique occurrences
    preliminary_cleaned_dataframe['nature_of_job'] = preliminary_cleaned_dataframe['nature_of_job'].apply(
        remove_repeats)
    save_dataframe(preliminary_cleaned_dataframe, 'preliminary_cleaned_dataframe.csv')
    print("Preliminary cleaning done successfully\n")

    # Step 2: Clean the textual columns
    columns_to_clean = [
        'requirements', 'benefits', 'type_of_contract', 'type_of_position',
        'required_education', 'nature_of_company', 'nature_of_job',
        'company_profile_and_description', 'title', 'function', 'description', 'company_profile'
    ]

    print("Cleaning the specified textual columns...")
    cleaner = Cleaner(preliminary_cleaned_dataframe)
    cleaned_dataframe = cleaner.perform_advanced_cleaning(columns_to_clean)
    save_dataframe(cleaned_dataframe, 'fully_cleaned_dataframe.csv')
    print("Textual columns cleaned successfully\n")

    # Step 3: Create the sentence embeddings. Should only be executed once
    if do_embedding:
        columns_to_embed = [
            'cleaned_requirements', 'cleaned_benefits', 'cleaned_type_of_contract', 'cleaned_type_of_position',
            'cleaned_required_education', 'cleaned_nature_of_company', 'cleaned_nature_of_job',
            'cleaned_company_profile_and_description', 'cleaned_title', 'cleaned_function', 'cleaned_description', 'cleaned_company_profile'
        ]
        embedding = Embedding(cleaned_dataframe)
        embedding.embed(columns_to_embed)


if __name__ == "__main__":
    main()
