import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set(style="whitegrid")

# 1) Load and inspect the dataset
df = pd.read_csv("customer_churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

print("\nData Info:")
df.info()

print("\nSummary Statistics:\n", df.describe())

print("\nChurn Distribution (counts):\n", df["Churn"].value_counts())
print("\nChurn Distribution (ratio):\n", df["Churn"].value_counts(normalize=True))

# 2) Check and handle missing values
print("\nMissing Values:\n", df.isnull().sum())

# Fill numeric columns with median and categorical with mode
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# 3) Exploratory Data Analysis
# Distribution of numeric variables
df.hist(figsize=(12, 10))
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Compare important features against churn
plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="Watch_Time_Hours", data=df)
plt.title("Watch Time Hours vs Churn")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="Number_of_Logins", data=df)
plt.title("Number of Logins vs Churn")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="Number_of_Complaints", data=df)
plt.title("Number of Complaints vs Churn")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="Churn", y="Resolution_Time_Days", data=df)
plt.title("Resolution Time vs Churn")
plt.tight_layout()
plt.show()

# Correlation between numeric features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# 4) Prepare data for modeling
# Remove ID column since it carries no predictive value
df = df.drop("CustomerID", axis=1)

# Convert categorical variables into numeric form
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Stratified split to preserve churn ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

# 5) Decision Tree (with tuning)
dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")

dt_params = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10]
}

dt_grid = GridSearchCV(dt, dt_params, cv=5)
dt_grid.fit(X_train, y_train)

print("\n=== Decision Tree Best Params ===")
print(dt_grid.best_params_)

y_pred_dt = dt_grid.predict(X_test)

print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

# Show top levels of the tree for interpretability
plt.figure(figsize=(22, 10))
plot_tree(
    dt_grid.best_estimator_,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    max_depth=3
)
plt.title("Decision Tree (Top Levels)")
plt.tight_layout()
plt.show()

# 6) Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# 7) Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== Top 10 Feature Importances ===")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# 8) Accuracy Comparison
print("\n=== Accuracy Comparison ===")
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))