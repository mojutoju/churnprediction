import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Load the synthetic dataset
data = pd.read_csv('synthetic_churn_data.csv')


# Subsample the data
sampled_data = data.sample(n=10000)  # Adjust the number of samples as needed

# Create a histogram
plt.hist(sampled_data['SubscriptionDuration'], bins=20)
plt.xlabel('Subscription Duration (months)')
plt.ylabel('Frequency')
plt.title('Distribution of Subscription Duration (Sampled)')
plt.show()


# Data Preprocessing
# 1. Encoding Categorical Variables
label_encoders = {}
categorical_columns = ['PaymentFrequency', 'UsageFrequency']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 2. Splitting Data into Features (X) and Target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']


# Summary Statistics
summary_stats = data.describe()

# Distribution of Churned vs. Retained Customers
churn_counts = data['Churn'].value_counts()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier (you can choose a different model)
model = RandomForestClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Visualize the confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.show()

feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.show()


# Distribution of Subscription Duration
plt.figure(figsize=(8, 6))
sns.histplot(data['SubscriptionDuration'], kde=True)
plt.xlabel('Subscription Duration (months)')
plt.ylabel('Frequency')
plt.title('Distribution of Subscription Duration')
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(data, hue='Churn', palette={0: 'blue', 1: 'red'})
plt.title('Pairplot of Features')
plt.show()

# Boxplots for Numeric Features
plt.figure(figsize=(12, 6))
sns.boxplot(x='PaymentFrequency', y='SubscriptionDuration', data=data)
plt.xlabel('Payment Frequency')
plt.ylabel('Subscription Duration')
plt.title('Boxplot of Subscription Duration by Payment Frequency')
plt.show()

# Barplots for Categorical Features
plt.figure(figsize=(10, 6))
sns.countplot(x='UsageFrequency', hue='Churn', data=data)
plt.xlabel('Usage Frequency')
plt.ylabel('Count')
plt.title('Churn Count by Usage Frequency')
plt.show()


