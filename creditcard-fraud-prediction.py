#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer


# In[29]:


#load the credit card fraud prediction data , source : Kaggle
df = pd.read_csv("/Users/pradyumna/pradyapps/bits/3rd-sem/data-science/assignment1/creditcard-fraud-data.csv")
df.head()


# In[7]:


#Information about the dataset
df.info()


# In[8]:


#Print count of rows and columns
print(f'The dataset contains {df.shape[0]} rows and {df.shape[1]} columns')


# In[124]:


# Print column names
print(df.columns.tolist())


# In[9]:


# Get data types of columns
column_types = df.dtypes

print(column_types)


# In[10]:


#Get unique values for each of the attributes
# Iterate over each column
for column in df.columns:
    # Get unique values for the current column
    unique_values = df[column].unique()
    print(f"Unique values for column {column}: {unique_values}")


# In[11]:


#Find mean , median and mode
selected_attributes = ['amt', 'zip', 'lat', 'city_pop', 'unix_time', 'merch_lat']
statistics = {}
for column in selected_attributes:
    if df[column].dtype != 'object':  # Exclude non-numeric columns
        mean_value = df[column].mean()
        median_value = df[column].median()
        mode_value = df[column].mode()[0]  # Mode can have multiple values, so we take the first one
        statistics[column] = {'mean': mean_value, 'median': median_value, 'mode': mode_value}

# Print statistics for each selected attribute
for column, values in statistics.items():
    print(f"Attribute: {column}")
    print(f"Mean: {values['mean']}")
    print(f"Median: {values['median']}")
    print(f"Mode: {values['mode']}")
    print()


# In[13]:


#Pearson Correlation Coefficients
# Select numeric columns
numeric_columns = ['cc_num', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 'is_fraud']
# Calculate Pearson correlation coefficients
correlation_matrix = df[numeric_columns].corr(method='pearson')
# Display the correlation matrix
print("Pearson Correlation Coefficients:")
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Coefficients')
plt.show()
print(correlation_matrix)


# In[22]:


# Check for missing values for each of the columns
missing_values = df.isnull().sum()
print("Missing values for each of the columns:")
print(missing_values)


# In[26]:


#Impute missing values
for column in df.columns:
    # Check if the column has missing values
    if df[column].isnull().any():
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Fill missing values with the mean of the column
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        else:
            # Fill missing values with the mode of the column
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
# Check again for missing values after imputation
missing_values_after = df.isnull().sum()
# Print the number of missing values after imputation
print("Number of missing values after imputation:")
print(missing_values_after)


# In[31]:


# Define a function to detect outliers using IQR
def detect_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers

# Iterate over each numeric column to check for outliers
numeric_columns = df.select_dtypes(include=['number']).columns
outliers_dict = {col: detect_outliers_iqr(df[col]) for col in numeric_columns if not detect_outliers_iqr(df[col]).empty}

# Print outliers for each numeric column
for col, outliers in outliers_dict.items():
    print(f"Outliers in column {col}:")
    print(outliers)
    print()


# In[28]:


# Check for duplicate rows
print(f"Number of duplicate rows in the DataFrame: {df.duplicated().sum()}")


# In[32]:


#Encoding Script - Label Encoding , OneHot Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Load the dataset
data = df
# Define binary categorical column
binary_cat_column = 'gender'
# Apply Label Encoding for binary categorical column - gender
label_encoder = LabelEncoder()
data[binary_cat_column] = label_encoder.fit_transform(data[binary_cat_column])

# Define multi-class categorical columns
multi_cat_columns = ['merchant', 'category']

# Apply One-Hot Encoding for multi-class categorical columns - merchant, category
one_hot_encoder = OneHotEncoder()
encoded_cols = one_hot_encoder.fit_transform(data[multi_cat_columns]).toarray()

# Get the feature names after encoding
feature_names = one_hot_encoder.get_feature_names_out(multi_cat_columns)

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_cols, columns=feature_names)

# Replace original categorical columns with encoded ones
data = pd.concat([data.drop(multi_cat_columns, axis=1), encoded_df], axis=1)

# Print the encoded dataset
print(data.head())


# In[36]:


#Scaling features 
from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the numeric columns
scaled_features = scaler.fit_transform(df[['amt', 'city_pop', 'unix_time']])

# Update the DataFrame with scaled features
df[['amt_scaled', 'city_pop_scaled', 'unix_time_scaled']] = scaled_features

# Display the scaled DataFrame
print(df)


# In[37]:


## Data Discretization Script

# Load the dataset
data = df

# Select columns to discretize
numeric_columns = ['amt', 'city_pop']

# Define the number of bins for each column
n_bins = 4  # You can adjust this based on your requirements

# Initialize KBinsDiscretizer with subsample=None to disable subsampling
kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)

# Fit and transform the selected columns
discretized_data = kbins.fit_transform(data[numeric_columns])

# Create a DataFrame for the discretized data
discretized_df = pd.DataFrame(discretized_data, columns=[f'{col}_bin' for col in numeric_columns])

# Replace original columns with discretized ones
data = pd.concat([data.drop(columns=numeric_columns), discretized_df], axis=1)

# Print the first few rows of the updated dataset
print(data.head())


# In[20]:


# Visualization for some of the attributes 
import matplotlib.pyplot as plt
import seaborn as sns

# Selecting data for visualization
selected_attributes = ['amt', 'zip', 'lat', 'city_pop']

# Scatter Plots
for column in selected_attributes:
    plt.figure(figsize=(8, 6))
    plt.scatter(df.index, df[column])
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.title(f'Scatter Plot for {column}')
    plt.show()

# Box Plots
for column in selected_attributes:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[column])
    plt.title(f'Box Plot for {column}')
    plt.show()

# Histograms
for column in selected_attributes:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=30, alpha=0.5)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram for {column}')
    plt.show()


# In[40]:


#Best 5 features script

# Load the dataset
data = df
# Preprocess the data
# Assuming 'trans_date_trans_time' and 'dob' are date/time columns that need to be converted
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
data['dob'] = pd.to_datetime(data['dob'], format='%d/%m/%Y')
# Extract relevant features from timestamp columns
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day'] = data['trans_date_trans_time'].dt.day
data['trans_month'] = data['trans_date_trans_time'].dt.month
data['dob_year'] = data['dob'].dt.year
# Encode categorical variables
encoder = LabelEncoder()
data['merchant'] = encoder.fit_transform(data['merchant'])
data['category'] = encoder.fit_transform(data['category'])
data['gender'] = encoder.fit_transform(data['gender'])
data['job'] = encoder.fit_transform(data['job'])

# Scale the data to ensure non-negative values
scaler = MinMaxScaler()
numeric_features = data.select_dtypes(include=['number']).columns.tolist()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Select only numeric features for feature selection
X = data[numeric_features].drop(['id', 'is_fraud', 'amt_scaled', 'unix_time_scaled'], axis=1)  # Exclude scaled columns
y = data['is_fraud']
# Select top 5 features using chi-squared test
best_features = SelectKBest(score_func=chi2, k=5)
fit = best_features.fit(X, y)
features_scores = pd.DataFrame({'Feature': X.columns, 'Score': fit.scores_})
top_features = features_scores.nlargest(5, 'Score')
# Display the top 5 features
print(top_features)


# In[ ]:




