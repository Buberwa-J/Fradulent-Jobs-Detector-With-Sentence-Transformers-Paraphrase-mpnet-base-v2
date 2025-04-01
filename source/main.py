# main.py
from source.classes.PreliminaryCleaner import PreliminaryCleaner
from source.paths import original_dataframe_path
from source.helpers import load_dataframe, save_dataframe


def main():
    original_dataframe = load_dataframe(original_dataframe_path)

    print("Starting the Job Legitimacy Detector Pipeline...\n")

    # Step 1: Preliminary Cleaning
    print("Step 1: Doing some preliminary cleaning on the data...")
    cleaner = PreliminaryCleaner(original_dataframe)
    preliminary_cleaned_dataframe = cleaner.clean_data()
    save_dataframe(preliminary_cleaned_dataframe, '../datasets', 'preliminary_cleaned_dataframe.csv')
    print("Completed Successfully")

    # Step 2:



if __name__ == "__main__":
    main()
