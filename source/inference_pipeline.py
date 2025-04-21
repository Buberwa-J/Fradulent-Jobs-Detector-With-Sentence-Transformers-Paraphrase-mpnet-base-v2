# inference_pipeline.py

import joblib
from source.classes.PreliminaryCleaner import PreliminaryCleaner
from source.classes.Cleaner import Cleaner
from source.classes.Embedding import Embedding
from source.helpers import save_dataframe, remove_repeats
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hgb_classifier_v1.joblib')


def run_inference(input_job_json):
    print("Starting the Job Legitimacy Inference Pipeline...\n")

    # Step 1: Load the new dataset
    df = pd.DataFrame([input_job_json])
    print("Data Loaded Successfully\n")

    # Step 2: Preliminary Cleaning
    print("Doing Preliminary Cleaning...\n")
    preliminary_cleaner = PreliminaryCleaner(df)
    df = preliminary_cleaner.clean_data()

    df['nature_of_job'] = df['nature_of_job'].apply(remove_repeats)
    print("Preliminary Cleaning Done\n")

    # Step 3: Clean textual columns
    columns_to_clean = [
        'requirements', 'benefits', 'type_of_contract', 'type_of_position',
        'required_education', 'nature_of_company', 'nature_of_job',
        'company_profile_and_description', 'title', 'function', 'description', 'company_profile'
    ]

    cleaner = Cleaner(df)
    df = cleaner.perform_advanced_cleaning(columns_to_clean)

    # Step 3: Embeddings
    columns_to_embed = [
        'cleaned_requirements', 'cleaned_benefits', 'cleaned_type_of_contract', 'cleaned_type_of_position',
        'cleaned_required_education', 'cleaned_nature_of_company', 'cleaned_nature_of_job',
        'cleaned_company_profile_and_description', 'cleaned_title', 'cleaned_function', 'cleaned_description',
        'cleaned_company_profile'
    ]

    embedding = Embedding(df)
    feature_df = embedding.embed_for_inference(columns_to_embed)

    # Step 4: Load trained model
    print(f"Loading model..\n")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded..\n")

    # Step 5: Predict
    predictions = model.predict(feature_df)
    probabilities = model.predict_proba(feature_df)[:, 1]  # Probability of being fraudulent

    # Step 6: Append predictions to original DataFrame
    df['fraudulent_prediction'] = predictions
    df['fraud_probability'] = probabilities

    print('Prediction Completed\n')

    return df


if __name__ == "__main__":

    test_json_fraud = {
        "title": "IC&E Technician",
        "company_profile": "Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â  Â Staffing & Recruiting done right for the Oil & Energy Industry! Represented candidates are automatically granted the following perks: Expert negotiations on your behalf, maximizing your compensation package and implimenting ongoing increases. Significant signing bonus by Refined Resources (in addition to any potential signing bonuses our client companies offer). 1 Year access to AnyPerk: significant corporate discounts on cell phones, event tickets, house cleaning and everything inbetween. Â You'll save thousands on daily expenditures. Professional Relocation Services for out of town candidates* All candidates are encouraged to participate in our Referral Bonus Program ranging anywhere from $500 - $1,000 for all successfully hired candidates... referred directly to the Refined Resources team. Please submit referrals via online Referral Form. Thank you and we look forward to working with you soon! Â [ Click to enlarge Image ]",
        "description": "IC&E Technician | Bakersfield, CA Mt. Poso. Principal Duties and Responsibilities: Calibrates, tests, maintains, troubleshoots, and installs all power plant instrumentation, control systems and electrical equipment. Performs maintenance on motor control centers, motor operated valves, generators, excitation equipment and motors. Performs preventive, predictive and corrective maintenance on equipment, coordinating work with various team members. Designs and installs new equipment and/or system modifications. Troubleshoots and performs maintenance on DC backup power equipment, process controls, programmable logic controls (PLC), and emission monitoring equipment. Uses maintenance reporting system to record time and material use, problem identified and corrected, and further action required; provides complete history of maintenance on equipment. Schedule, coordinate, work with and monitor contractors on specific tasks, as required. Follows safe working practices at all times. Identifies safety hazards and recommends solutions. Follows environmental compliance work practices. Identifies environmental non-compliance problems and assist in implementing solutions. Assists other team members and works with all departments to support generating station in achieving their performance goals. Trains other team members in the areas of instrumentation, control, and electrical systems. Performs housekeeping assignments, as directed. Conduct equipment and system tagging according to company and plant rules and regulations. Perform equipment safety inspections, as required, and record results as appropriate. Participate in small construction projects. Read and interpret drawings, sketches, prints, and specifications, as required. Orders parts as needed to affect maintenance and repair. Performs Operations tasks on an as-needed basis and other tasks as assigned. Available within a reasonable response time for emergency call-ins and overtime, plus provide acceptable off-hour contact by phone and company pager. Excellent Verbal and Written Communications Skills: Ability to coordinate work activities with other team members on technical subjects across job families. Ability to work weekends, holidays, and rotating shifts, as required.",
        "requirements": "Qualifications. Knowledge, Skills & Abilities: A high school diploma or GED is required. Must have a valid driver's license. Ability to read, write, and communicate effectively in English. Good math skills. Four years of experience as an I&C Technician and/or Electrician in a power plant environment, preferably with a strong electrical background, up to and including, voltages to 15 KV to provide the following: Demonstrated knowledge of electrical equipment, electronics, schematics, basics of chemistry and physics and controls and instrumentation. Demonstrated knowledge of safe work practices associated with a power plant environment. Demonstrated ability to calibrate I&C systems and equipment, including analytic equipment. Demonstrated ability to configure and operate various test instruments and equipment, as necessary, to troubleshoot and repair plant equipment including, but not limited to, distributed control systems, programmable logic controllers, motor control centers, transformers, generators, and continuous emissions monitor (CEM) systems. Demonstrated ability to work with others in a team environment.",
        "benefits": "What is offered: Competitive compensation package. 100% matched retirement fund. Annual vacations paid for by company. Significant bonus structure. Opportunity for advancement. Full benefits package. Annual performance reviews and base salary increases. Annual cost of living increases. Sound, clean, safe and enjoyable working environment & Company Culture. World renowned management and executive team who promote from within, leverage careers and invest in employees for the long-term success of their careers and overall company/employee goals. Qualified candidates contact: Darren Lawson | VP of Recruiting | #EMAIL_395225df8eed70288fc67310349d63d49d5f2ca6bc14dbb5dcbf9296069ad88c# | #PHONE_70128aad0c118273b0c2198a08d528591b932924e165b6a8d1272a6f9e2763d1#",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "High School or equivalent",
        "industry": "Oil & Energy",
        "function": "Other",
    }

    run_inference(test_json_fraud)