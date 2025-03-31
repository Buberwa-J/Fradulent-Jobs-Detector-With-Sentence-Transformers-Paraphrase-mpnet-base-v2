import pandas as pd


# # A little cleanup for the 'salary_range' column
# 1. The <u>uncleaned df </u> has the salary_range column that is cleaner than that of the <u> semi cleaned df </u>
# 2. Extract the minimum and the maximum salary

# In[178]:


# Step 1: Use the column from the uncleaned dataset
df['salary_range'] = uncleaned_df['salary_range']


# Step 2: Function to extract min and max values and return as floats
def extract_min_max(salary):
    try:
        # Split the range into min and max values
        low, high = map(float, salary.split('-'))
        return f"{low:.1f}", f"{high:.1f}"  # Format as readable floats
    except (ValueError, AttributeError):
        # Handle cases where extraction is impossible "Not Mentioned' also falls under this categ
        return None, None


# Apply the function to extract and format salary_min and salary_max
df[['salary_min', 'salary_max']] = df['salary_range'].apply(lambda x: pd.Series(extract_min_max(x)))


# Step 3: Define a function to determine the type of salary
def determine_salary_type(salary_max):
    try:
        # Convert salary_max to a float if it's not already
        salary = float(salary_max)

        # Flags for different salary types
        # These ranges are very subjective. I could definately do some more digging
        is_hourly = 0 < salary < 500.0
        is_monthly = 500.0 <= salary <= 30000.0
        is_annual = salary > 30000.0

        return is_hourly, is_monthly, is_annual
    except (ValueError, TypeError):
        # Handle cases where salary_max is invalid or missing
        return False, False, False


# Apply the function and unpack the results into new columns
df[['is_hourly', 'is_monthly', 'is_annual']] = df['salary_max'].apply(
    lambda x: pd.Series(determine_salary_type(x))
)

# Ensure the extracted salary columns are of float type (in case they were not properly converted)
df['salary_min'] = df['salary_min'].astype('float32')
df['salary_max'] = df['salary_max'].astype('float32')

# In[180]:


df.required_education.unique()

# # Engineer some features from the 'required_education' column.
# 1. Since I have few unique values, I could just easily create a dictionary with maps. These values are also subjective so I could definately dig abit deeper

# In[183]:


education_bins = {
    'unspecified': np.nan,
    'high school or equivalent': 2,
    'some high school coursework': 2,
    'vocational - hs diploma': 2,
    'vocational': 3,
    'vocational - degree': 3,
    'certification': 3,
    'some college coursework completed': 4,
    'associate degree': 4,
    "bachelor's degree": 5,
    'professional': 5,
    "master's degree": 7,
    "doctorate": 8,
}

df['required_education_numeric'] = df['required_education'].map(education_bins).astype('float16')

# In[185]:


df.type_of_position.unique()

# # Engineer some features from the 'type_of_position' column.
# 1. Since I have few unique values, I could just easily create a dictionary with maps just like in the education. These values are also subjective so I could definately dig abit deeper

# In[188]:


# Define the mapping
position_mapping = {
    'not applicable': np.nan,
    'internship': 2,
    'entry level': 3,
    'associate': 3,
    'mid-senior level': 4,
    'director': 5,
    'executive': 6
}

# Map the values
df['type_of_position_numeric'] = df['type_of_position'].map(position_mapping).astype('float16')

# In[190]:


df.type_of_contract.unique()

# # Create some encoding for the type_of_contract feature. I've chosen one-hot since I don't want to risk creating something ordinal where there shouldn't be anything ordinal

# In[193]:


# Perform one-hot encoding
df = pd.get_dummies(df, columns=['type_of_contract'], drop_first=True)

# # Create new features based on the minimum salary mentioned
# 1. Product of minimum salary and required_education_numeric
# 2. Product of minimum salary and type_of_position_numeric

# In[196]:


# Salary * Education
df['salary_education_min_product'] = (df['salary_min'] * df['required_education_numeric'])
df['salary_education_max_product'] = (df['salary_max'] * df['required_education_numeric'])

# Salary * position
df['salary_position_min_product'] = (df['salary_min'] * df['type_of_position_numeric'])
df['salary_position_max_product'] = (df['salary_max'] * df['type_of_position_numeric'])