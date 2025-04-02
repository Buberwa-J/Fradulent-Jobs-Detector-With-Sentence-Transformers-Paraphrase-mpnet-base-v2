import pandas as pd
import os




class PreliminaryCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self) -> pd.DataFrame:
        """Cleans the dataset by removing unnecessary columns and normalizing text"""

        self.df = self.df.drop(columns=['job_id', 'location', 'department',
                                        'telecommuting', 'has_company_logo', 'has_questions'])

        self.df['nature_of_job'] = self.df['title'] + ' ' + self.df['function']
        self.df.drop(columns=['title', 'function'], inplace=True)

        self.df.rename(columns={'industry': 'nature_of_company',
                                'required_experience': 'type_of_position',
                                'employment_type': 'type_of_contract'}, inplace=True)

        self.df[self.df.select_dtypes(include='object').columns] = self.df.select_dtypes(include='object').apply(
            lambda x: x.str.lower())

        return self.df

