import pandas as pd
import numpy as np
# Loading merged data
df = pd.read_csv('E:/summer training/final_project/merged_data.csv')

# Fix date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop NaT rows if needed
df = df.dropna(subset=['Date'])

#2. Matplotlib & Seaborn Plots
#a. Line Chart: Closing Price Over Time (for a coin)

import matplotlib.pyplot as plt
import seaborn as sns

btc = df[df['Coin_Name'] == 'Bitcoin']

plt.figure(figsize=(12, 5))
sns.lineplot(x='Date', y='Close', data=btc)
plt.title('Bitcoin Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.tight_layout()
plt.show()

# b. Bar Plot: Top 10 Coins by Avg Market Cap
avg_mcap = df.groupby('Coin_Name')['Marketcap'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
avg_mcap.plot(kind='bar', color='skyblue')
plt.title('Top 10 Coins by Avg. Market Cap')
plt.ylabel('Avg Market Cap')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# c. Scatter Plot: % Change vs Volatility
# Filter data (e.g., keep between -50% to +50% change and volatility < 1000)
filtered_df = df[(df['Percent_Change'].between(-50, 50)) & (df['Volatility'] < 1000)]

plt.figure(figsize=(12, 6))
sns.scatterplot(data=filtered_df, x='Percent_Change', y='Volatility', hue='Coin_Name', alpha=0.6)
plt.title('% Change vs Volatility (Filtered View)')
plt.xlabel('% Change')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()


#d. Pie Plot: Current Market Cap Share (latest date)
latest_date = df['Date'].max()
latest = df[df['Date'] == latest_date]
top10 = latest.sort_values('Marketcap', ascending=False).head(10)

plt.figure(figsize=(8, 8))
plt.pie(top10['Marketcap'], labels=top10['Coin_Name'], autopct='%1.1f%%')
plt.title('Market Cap Share on ' + str(latest_date.date()))
plt.show()

#  3. ML: KNN and Decision Tree on Classification Task
df['Target'] = df['Percent_Change'].apply(lambda x: 1 if x > 0 else 0)  # 1 = gain, 0 = loss

#Preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = ['Open', 'High', 'Low', 'Close', 'Volatility', 'Marketcap']
X = df[features]
y = df['Target']

# Handle infinite/missing
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#a. KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm_knn = confusion_matrix(y_test, y_pred)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Loss", "Gain"])
disp_knn.plot(cmap='Blues')
plt.title("KNN Confusion Matrix")
plt.show()

# b. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["Loss", "Gain"])
disp_dt.plot(cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.show()

#Feature Importance (Decision Tree)
importances = dt.feature_importances_
for feat, score in zip(features, importances):
    print(f"{feat}: {score:.4f}")

features = ['Open', 'High', 'Low', 'Close', 'Volatility', 'Marketcap']
importances = dt.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Decision Tree Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()