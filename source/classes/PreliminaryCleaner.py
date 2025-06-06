import pandas as pd


class PreliminaryCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        """Cleans the dataset by removing unnecessary columns and normalizing text"""
        self.df['nature_of_job'] = self.df['title'] + ' ' + self.df['function']
        self.df['company_profile_and_description'] = self.df['company_profile'] + ' ' + self.df['description']

        # You could un comment this for the training pipeline but for inference only needed data is collected from the user
        # self.df.drop(columns=['title', 'function', 'company_profile', 'description', 'job_id', 'location', 'department',
        #                       'telecommuting', 'has_company_logo', 'has_questions'], inplace=True)

        self.df.rename(columns={'industry': 'nature_of_company',
                                'required_experience': 'type_of_position',
                                'employment_type': 'type_of_contract'}, inplace=True)

        self.df[self.df.select_dtypes(include='object').columns] = self.df.select_dtypes(include='object').apply(
            lambda x: x.str.lower())

        return self.df
