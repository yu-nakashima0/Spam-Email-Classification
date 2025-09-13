import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


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


#most common words in spam/not spam messages
word_features = data.iloc[:, 0:48]
word_counts = word_features.sum(axis=1)
plt.figure(figsize=(10,6))
sns.barplot(data= word_counts)


#Compare average message length between spam and ham
avg_len_spam = data[data["class"] == 1]["capital_run_length_total"].mean()
avg_len_notspam = data[data["class"] == 0]["capital_run_length_total"].mean()
print(f"Avarage length of spam messages: {avg_len_spam}")
print(f"Avarage length of not spam messages: {avg_len_notspam}")


#split data in train/test data
X = data.drop(columns="class")
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 42
)


#logistic  regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


#random forest model
randomforest_model = RandomForestClassifier(n_estimators=10, random_state=42)
randomforest_model.fit(X_train,y_train)
y_pred = randomforest_model.predict(X_test)
print("Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))