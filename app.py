import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load the dataset
df = pd.read_csv("./spambase_csv.csv")


#the info about the dataset 
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)


#Correlation of each feature with other features
corr = df.corr(numeric_only=True)
print(corr)


#Heatmap to visualize the correlation
plt.figure(figsize=(8,6))
sns.heatmap(data=corr, annot = False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


#number of dublicate data and removing duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print(f"Shape before removing duplicates: {df.shape}")
data = df.drop_duplicates()
print(f"Shape after removing duplicates: {data.shape}")


#number of null values 
num_null = data.isnull().sum()
print(f"Number of null values in each column: {num_null}")



