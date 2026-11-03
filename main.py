import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('insurance.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
print(df.shape)

# #Exploratory Data Analysis(EDA)
# print(df.isnull().sum())

numeric_columns = ['age', 'bmi', 'children', 'charges']

# plt.figure(figsize=(10, 5))
# sns.countplot(x = df['sex'])
# plt.show()

# sns.countplot(x = df['smoker'])
# plt.show()

# sns.countplot(x = df['region'])
# plt.show()

# for col in numeric_columns:
#     plt.figure(figsize=(10, 5))
#     sns.histplot(df[col], kde=True, bins=20)
#     plt.show()

# for col in numeric_columns:
#     plt.figure(figsize=(10, 5))
#     sns.boxplot(x = df[col])
#     plt.show()

# plt.figure(figsize=(10, 5))
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

#Data Cleaning and Preprocessing

df_cleaned = df.copy()

df_cleaned.drop_duplicates(inplace=True)
# print(df_cleaned.shape)

# print(df_cleaned['sex'].value_counts())

#Encoding categorical variables

df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
# print(df_cleaned.head())

df_cleaned['smoker'] = df_cleaned['smoker'].map({'no':0,'yes':1})
# print(df_cleaned.head())

df_cleaned.rename(columns={'sex':'is_female', 'smoker':'is_smoker'}, inplace=True)
# print(df_cleaned.head())

# print(df_cleaned['region'].value_counts())

#One-hot encoding for 'region' column

df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
df_cleaned = df_cleaned.astype(int, errors='ignore')
# print(df_cleaned.head())

#Feature Engineering 

df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# print(df_cleaned.head())

df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)

df_cleaned = df_cleaned.astype(int, errors='ignore')
# print(df_cleaned.head())

from sklearn.preprocessing import StandardScaler

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()

df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

# print(df_cleaned.head())

#Feature Extraction

from scipy.stats import pearsonr, chi2_contingency

#Pearson Correlation Calculation

selected_features = ['age', 'bmi', 'children', 'is_female', 'is_smoker', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese']

correlations = {feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0] for feature in selected_features}

df_correlations = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Pearson Correlation'])
df_correlations = df_correlations.sort_values(by='Pearson Correlation', ascending=False)
# print(df_correlations)

cat_features = ['is_female', 'is_smoker', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese']

alpha = 0.05
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)
chi2_results = {}

for feature in cat_features:
    contingency= pd.crosstab(df_cleaned[feature], df_cleaned['charges_bin'])
    chi2_stat, p_val,_ ,_= chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'
    chi2_results[feature] = {'Chi2_Statistic': chi2_stat, 'p_value': p_val, 'Decision': decision}

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')
print(chi2_df)

final_df = df_cleaned[['age','is_female','bmi','children','is_smoker','region_southeast','bmi_category_Obese','charges']]
print(final_df.head())