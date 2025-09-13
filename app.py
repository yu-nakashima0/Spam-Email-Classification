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



#Word counts distribution per message
word_features = data.iloc[:, 0:48]
word_counts = word_features.sum(axis=1)
plt.figure(figsize=(10,6))
sns.histplot(word_counts[data["class"]==0], bins=50, color="skyblue", label="Not Spam", kde=True)
sns.histplot(word_counts[data["class"]==1], bins=50, color="salmon", label="Spam", kde=True)
plt.title("Word Count Distribution per Message (Spam vs Not Spam)")
plt.xlabel("Estimated Word Count")
plt.ylabel("Number of Messages")
plt.legend()
plt.show()


#message length distribution per message
plt.figure(figsize=(10,6))
sns.histplot(data=data[data["class"]==0]["capital_run_length_total"], 
             bins=70, color="skyblue", label="Not Spam", kde=True)
sns.histplot(data=data[data["class"]==1]["capital_run_length_total"], 
             bins=70, color="salmon", label="Spam", kde=True)
plt.title("Message Length Distribution per Message (Spam vs Not Spam)")
plt.xlabel("length of Message")
plt.ylabel("Number of Messages")
plt.legend()
plt.show()