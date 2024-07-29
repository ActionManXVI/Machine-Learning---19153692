# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.le_country = LabelEncoder()
        self.le_store_class = LabelEncoder()
        
    def preprocess(self):
        # Handle missing values
        self.df = self.df.dropna()
        
        # Merge store class "F" with "Flagship"
        self.df['storeClass'] = self.df['storeClass'].replace('F', 'Flagship')
        
        # Extract relevant features
        self.features = ['longitude', 'latitude', 'storeClass', 'country']
        for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
            self.df[f'{day}_hours'] = self.df[f'{day}_open_hours'].apply(lambda x: int(x.split('-')[1].split(':')[0]) - int(x.split('-')[0].split(':')[0]))
        self.features.extend([f'{day}_hours' for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']])
        
        # Encode categorical variables
        self.df['country_encoded'] = self.le_country.fit_transform(self.df['country'])
        self.df['storeClass_encoded'] = self.le_store_class.fit_transform(self.df['storeClass'])
        
        # New feature: Total weekly hours
        self.df['total_weekly_hours'] = self.df[[f'{day}_hours' for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]].sum(axis=1)
        
        # New feature: Weekend ratio
        self.df['weekend_ratio'] = (self.df['Sat_hours'] + self.df['Sun_hours']) / self.df['total_weekly_hours']
        
        # New feature: Is capital city (example for a few countries)
        capitals = {'United States': 'Washington', 'United Kingdom': 'London', 'France': 'Paris', 'Germany': 'Berlin'}
        self.df['is_capital'] = self.df.apply(lambda row: int(row['city'] == capitals.get(row['country'], '')), axis=1)
        
        # New feature: Approximate population (this would require an external dataset)
        # For demonstration, we'll use a dummy function
        self.df['approx_population'] = self.df.apply(self._get_approx_population, axis=1)
        
        self.features.extend(['total_weekly_hours', 'weekend_ratio', 'is_capital', 'approx_population'])
        
        return self.df
    
    def _get_approx_population(self, row):
        # This is a dummy function. In reality, you would use a dataset of city populations
        # or an API to get this information
        return np.random.randint(10000, 1000000)
    
    def get_cluster_data(self):
        return self.df[['latitude', 'longitude']]
    
    def get_country_prediction_data(self):
        X = self.df[['latitude', 'longitude', 'storeClass_encoded', 'total_weekly_hours', 'weekend_ratio', 'is_capital', 'approx_population'] + 
                    [f'{day}_hours' for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]]
        y = self.df['country_encoded']
        return X, y
    
    def get_store_classification_data(self):
        X = self.df[['latitude', 'longitude', 'country_encoded', 'total_weekly_hours', 'weekend_ratio', 'is_capital', 'approx_population'] + 
                    [f'{day}_hours' for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']]]
        y = self.df['storeClass_encoded']
        return X, y
    
    def get_country_names(self):
        return self.le_country.classes_
    
    def get_store_class_names(self):
        return self.le_store_class.classes_
