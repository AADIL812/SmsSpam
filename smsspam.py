import numpy as np
import pandas as pd


df = pd.read_csv('spam.csv',encoding='latin-1')

#print(df.sample(5))


#1. Data cleaning
#2. EDA (Exploratary Data Analysis)
#3. Text preprocessing
#4. Model Building
#5. Model Evaluation
#6. Improve the model
#7. Website
#8. Deploy 


##1. Data Cleaning 

# drop last 3 columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
print(df.sample(5))

# rename columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
print(df.sample(5))

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

df['target']=encoder.fit_transform(df['target'])
print(df.head())


#check for missing values
print(df.isnull().sum())

#check for duplicate values

print(df.duplicated().sum())

df=df.drop_duplicates(keep='first')
print(df.duplicated().sum())



## 2. EDA (Exploratory Data Analysis)

print(df['target'].value_counts())

import matplotlib.pyplot as plt

plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()
# data is imbalanced as there is only very few samples for spam

import nltk
nltk.download('punkt')


