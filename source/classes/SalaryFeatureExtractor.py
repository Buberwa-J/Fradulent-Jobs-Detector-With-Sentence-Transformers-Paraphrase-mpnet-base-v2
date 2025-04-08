import pandas as pd


# Step 1: Function to extract min and max values and return as floats
def extract_min_max(salary):
    try:
        # Split the range into min and max values
        low, high = map(float, salary.split('-'))
        return f"{low:.1f}", f"{high:.1f}"  # Format as readable floats
    except (ValueError, AttributeError):
        # Handle cases where extraction is impossible "Not Mentioned" also falls under this category
        return None, None


# Step 2: Define a function to determine the type of salary
def determine_salary_type(salary_max):
    try:
        # Convert salary_max to a float if it's not already
        salary = float(salary_max)

        # Flags for different salary types
        # These ranges are very subjective. I could definitely do some more digging
        is_hourly = 0 < salary < 500.0
        is_monthly = 500.0 <= salary <= 30000.0
        is_annual = salary > 30000.0

        return is_hourly, is_monthly, is_annual
    except (ValueError, TypeError):
        # Handle cases where salary_max is invalid or missing
        return False, False, False


class SalaryFeatureExtractor:
    def __init__(self, df):
        self.df = df

    def process(self):
        # Create a new DataFrame to hold only the extracted features related to salary
        extracted_features_df = pd.DataFrame()

        # Extract the maximum and minimum salaries
        extracted_features_df[['salary_min', 'salary_max']] = self.df['salary_range'].apply(lambda x: pd.Series(extract_min_max(x)))

        # Apply the function and unpack the results into new columns
        extracted_features_df[['is_hourly', 'is_monthly', 'is_annual']] = extracted_features_df['salary_max'].apply(
            lambda x: pd.Series(determine_salary_type(x))
        )

        # Ensure the extracted salary columns are of float type (in case they were not properly converted)
        extracted_features_df['salary_min'] = extracted_features_df['salary_min'].astype('float32')
        extracted_features_df['salary_max'] = extracted_features_df['salary_max'].astype('float32')

        return extracted_features_df



