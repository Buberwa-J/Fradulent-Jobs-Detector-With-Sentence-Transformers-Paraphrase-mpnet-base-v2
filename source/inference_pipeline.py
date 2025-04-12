import joblib
import pandas as pd
from source.classes.PreliminaryCleaner import PreliminaryCleaner
from source.classes.Cleaner import Cleaner
from source.classes.Embedding import Embedding
from source.helpers import save_dataframe, remove_repeats
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(input_job_json):
    """Load the job data into a pandas DataFrame."""
    try:
        df = pd.DataFrame([input_job_json])
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Perform preliminary and advanced cleaning on the dataset."""
    try:
        logger.info("Starting preliminary cleaning...")
        preliminary_cleaner = PreliminaryCleaner(df)
        df = preliminary_cleaner.clean_data()
        df['nature_of_job'] = df['nature_of_job'].apply(remove_repeats)
        logger.info("Preliminary cleaning completed.")

        columns_to_clean = [
            'requirements', 'benefits', 'type_of_contract', 'type_of_position',
            'required_education', 'nature_of_company', 'nature_of_job',
            'company_profile_and_description', 'title', 'function', 'description', 'company_profile'
        ]

        cleaner = Cleaner(df)
        df = cleaner.perform_advanced_cleaning(columns_to_clean)
        logger.info("Advanced cleaning completed.")
        return df
    except Exception as e:
        logger.error(f"Error during cleaning: {e}")
        raise

def generate_embeddings(df):
    """Generate embeddings for the relevant columns."""
    try:
        columns_to_embed = [
            'cleaned_requirements', 'cleaned_benefits', 'cleaned_type_of_contract', 'cleaned_type_of_position',
            'cleaned_required_education', 'cleaned_nature_of_company', 'cleaned_nature_of_job',
            'cleaned_company_profile_and_description', 'cleaned_title', 'cleaned_function', 'cleaned_description',
            'cleaned_company_profile'
        ]

        embedding = Embedding(df)
        feature_df = embedding.embed_for_inference(columns_to_embed)
        logger.info("Embeddings generated successfully.")
        return feature_df
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def load_model(model_path='models/hgb_classifier_v1.joblib'):
    """Load the pre-trained model."""
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def make_predictions(model, feature_df):
    """Make predictions and calculate probabilities."""
    try:
        predictions = model.predict(feature_df)
        probabilities = model.predict_proba(feature_df)[:, 1]  # Probability of being fraudulent
        logger.info("Predictions and probabilities computed.")
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def append_predictions_to_df(df, predictions, probabilities):
    """Append the predictions and probabilities to the original DataFrame."""
    df['fraudulent_prediction'] = predictions
    df['fraud_probability'] = probabilities
    logger.info("Predictions appended to DataFrame.")
    return df

def save_results(df, output_file='output_df.csv'):
    """Save the results to a CSV file."""
    try:
        save_dataframe(df, output_file)
        logger.info(f"Results saved to {output_file}.")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def run_inference(input_job_json):
    """Run the full job legitimacy inference pipeline."""
    logger.info("Starting the Job Legitimacy Inference Pipeline...\n")

    # Step 1: Load Data
    df = load_data(input_job_json)

    # Step 2: Clean Data
    df = clean_data(df)

    # Step 3: Generate Embeddings
    feature_df = generate_embeddings(df)

    # Step 4: Load Model
    model = load_model()

    # Step 5: Make Predictions
    predictions, probabilities = make_predictions(model, feature_df)

    # Step 6: Append Predictions to DataFrame
    df = append_predictions_to_df(df, predictions, probabilities)

    # Step 7: Save Results (optional for production)
    save_results(df)

    logger.info("Inference pipeline completed successfully.")

if __name__ == "__main__":
    test_json_fake_2 = {
        "requirements": "No experience necessary! Work from home and earn huge amounts of money in no time. Must have a computer and internet access. We offer fast training.",
        "benefits": "Unlimited income potential, bonuses, work-from-home flexibility, and be your own boss. Get paid weekly.",
        "type_of_contract": "Freelance/Contract (Unspecified)",
        "type_of_position": "Remote Work-from-Home Money-Making Opportunity",
        "required_education": "No formal education required, just a willingness to learn.",
        "nature_of_company": "An online company offering quick financial returns through simple online tasks.",
        "nature_of_job": "Earn money by doing small online tasks like surveys, simple data entry, or posting ads online. Set your own hours and work as much as you want!",
        "company_profile_and_description": "FastCash Network is a rapidly growing company offering people the chance to earn money from home through simple online tasks. No experience necessary. Work as much or as little as you want.",
        "title": "Earn $1000 a Day with No Experience Required!",
        "function": "Data entry, online marketing, ad posting, and survey participation",
        "description": "This is a fast-paced and exciting opportunity to make quick cash by completing simple online tasks. No training required! Just follow easy instructions and start earning money within hours. Full-time and part-time options available.",
        "company_profile": "FastCash Network is an innovative company that connects people with money-making opportunities. We help people earn money from the comfort of their homes by completing easy online jobs."
    }

    run_inference(test_json_fake_2)
