# combined_store_analysis_script.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from DataPreprocessing import DataPreprocessor 

# Set the working directory
os.chdir(os.path.dirname(__file__))

# Run the data preprocessor
preprocessor = DataPreprocessor('hm_all_stores.csv')
df = preprocessor.preprocess()

# Calculate total weekly hours
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df['total_weekly_hours'] = df[[f'{day}_hours' for day in days]].sum(axis=1)

# Save the preprocessed data to a new CSV file
df.to_csv('hm_store_data_preprocessed.csv', index=False)



# Load the preprocessed data
df = pd.read_csv('hm_store_data_preprocessed.csv')
df['storeClass'] = df['storeClass'].astype(str).replace({'F': 'Flagship', 'nan': 'N/A'})


# 1. Basic Statistical Summary
print(df.describe())

# 2. Distribution of Store Classes
plt.figure(figsize=(10, 6))
df['storeClass'].value_counts().plot(kind='bar')
plt.title('Distribution of Store Classes')
plt.ylabel('Count')
plt.show()

# 3. Geographical Distribution of Stores
plt.figure(figsize=(12, 8))
plt.scatter(df['longitude'], df['latitude'], alpha=0.5)
plt.title('Geographical Distribution of H&M Stores')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# 4. Distribution of Total Weekly Hours
plt.figure(figsize=(10, 6))
sns.histplot(df['total_weekly_hours'], kde=True)
plt.title('Distribution of Total Weekly Hours')
plt.xlabel('Hours')
plt.show()

# 5. Correlation Heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 6. Opening Hours Patterns
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
avg_hours = [df[f'{day}_hours'].mean() for day in days]

plt.figure(figsize=(10, 6))
plt.bar(days, avg_hours)
plt.title('Average Opening Hours by Day of Week')
plt.ylabel('Average Hours')
plt.show()

# 7. Store Class vs Total Weekly Hours
plt.figure(figsize=(10, 6))
df.groupby('storeClass')['total_weekly_hours'].mean().plot(kind='bar')
plt.title('Store Class vs Total Weekly Hours')
plt.ylabel('Average Total Weekly Hours')
plt.xlabel('Store Class')
plt.show()

# 8. Countries with Most Stores
top_countries = df['country'].value_counts().head(10)
plt.figure(figsize=(12, 6))
top_countries.plot(kind='bar')
plt.title('Top 10 Countries by Number of Stores')
plt.xlabel('Country')
plt.ylabel('Number of Stores')
plt.show()

# 9. Timezone Distribution
plt.figure(figsize=(10, 6))
df['timeZoneIndex'].hist(bins=30)
plt.title('Distribution of Stores Across Time Zones')
plt.xlabel('Time Zone Index')
plt.ylabel('Number of Stores')
plt.show()
