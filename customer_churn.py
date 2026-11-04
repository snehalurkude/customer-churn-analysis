
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)

# 1. Load the data

df = pd.read_csv("data/customer_data.csv")

print("Shape of data:", df.shape)
print(df.head())
print()
df.info()
print()
print("Missing values in each column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


# 2. Data cleaning

# remove duplicate rows
df = df.drop_duplicates()

# total_charges was read as text because some rows have a blank space.
# convert it to numbers, blanks become NaN
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

# fill missing numbers with the median of the column
for col in ["tenure", "monthly_charges", "total_charges"]:
    df[col] = df[col].fillna(df[col].median())

# churn is Yes/No, change it to 1/0 so the model can use it
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

print("\nMissing values after cleaning:", df.isnull().sum().sum())
print("Shape after cleaning:", df.shape)


# 3. Exploratory data analysis

churn_rate = df["churn"].mean() * 100
print("\nOverall churn rate: {:.1f}%".format(churn_rate))
print(df["churn"].value_counts())

print("\nChurn rate by contract type (%):")
print((df.groupby("contract")["churn"].mean() * 100).round(1))

print("\nChurn rate by internet service (%):")
print((df.groupby("internet_service")["churn"].mean() * 100).round(1))

print("\nChurn rate by payment method (%):")
print((df.groupby("payment_method")["churn"].mean() * 100).round(1))

print("\nAverage values for customers who stayed (0) vs left (1):")
print(df.groupby("churn")[["tenure", "monthly_charges"]].mean().round(2))

# plot 1 - how many customers left vs stayed
counts = df["churn"].value_counts().sort_index()
plt.figure(figsize=(5, 4))
plt.bar(["Stayed", "Left"], counts.values, color=["steelblue", "tomato"])
plt.title("Customers Who Stayed vs Left")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig("outputs/churn_count.png")
plt.show()

# plot 2 - churn rate by contract type
rate = df.groupby("contract")["churn"].mean() * 100
plt.figure(figsize=(6, 4))
plt.bar(rate.index, rate.values, color="orange", edgecolor="black")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate (%)")
plt.tight_layout()
plt.savefig("outputs/churn_by_contract.png")
plt.show()

# plot 3 - churn rate by internet service
rate = df.groupby("internet_service")["churn"].mean() * 100
plt.figure(figsize=(6, 4))
plt.bar(rate.index, rate.values, color="mediumpurple", edgecolor="black")
plt.title("Churn Rate by Internet Service")
plt.ylabel("Churn rate (%)")
plt.tight_layout()
plt.savefig("outputs/churn_by_internet.png")
plt.show()

# plot 4 - tenure of customers who stayed vs left
plt.figure(figsize=(6, 4))
plt.hist(df[df["churn"] == 0]["tenure"], bins=15, alpha=0.6, label="Stayed", color="steelblue")
plt.hist(df[df["churn"] == 1]["tenure"], bins=15, alpha=0.6, label="Left", color="tomato")
plt.title("Tenure of Customers (Stayed vs Left)")
plt.xlabel("Tenure (months)")
plt.ylabel("Number of customers")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/tenure_vs_churn.png")
plt.show()

# plot 5 - monthly charges of customers who stayed vs left
plt.figure(figsize=(6, 4))
plt.boxplot([df[df["churn"] == 0]["monthly_charges"], df[df["churn"] == 1]["monthly_charges"]])
plt.xticks([1, 2], ["Stayed", "Left"])
plt.title("Monthly Charges: Stayed vs Left")
plt.ylabel("Monthly charges")
plt.tight_layout()
plt.savefig("outputs/charges_vs_churn.png")
plt.show()


# 4. Prepare features and target

# customer_id is just a label, so it is not used for prediction
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

# text columns are converted into 0/1 columns (one-hot encoding)
X = pd.get_dummies(X, columns=["gender", "contract", "internet_service", "payment_method"],
                   drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", len(X_train), "| Testing rows:", len(X_test))

# scale the numbers so that big values (like total_charges) don't dominate
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model

# only about 1 in 4 customers leave, so the model can ignore them and still look accurate.
# class_weight="balanced" makes it pay more attention to the customers who leave
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)


# 6. Evaluate the model

y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

# which features matter most for the model
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("Features pushing towards churn (highest positive values):")
print(coef.tail(3).round(2))
print("Features pushing towards staying (most negative values):")
print(coef.head(3).round(2))

# plot 6 - feature importance from the model
plt.figure(figsize=(7, 5))
colors = ["tomato" if v > 0 else "steelblue" for v in coef.values]
plt.barh(coef.index, coef.values, color=colors)
plt.title("Effect of Each Feature on Churn")
plt.xlabel("Model coefficient (positive = more likely to leave)")
plt.tight_layout()
plt.savefig("outputs/feature_effects.png")
plt.show()


# 7. Try the model on a new customer

new_customer = pd.DataFrame([{
    "senior_citizen": 0, "tenure": 5, "monthly_charges": 85.0, "total_charges": 425.0,
    "gender_Male": 1, "contract_One year": 0, "contract_Two year": 0,
    "internet_service_Fiber optic": 1, "internet_service_No": 0,
    "payment_method_Credit card": 0, "payment_method_Electronic check": 1,
    "payment_method_Mailed check": 0
}])
new_customer = new_customer[X.columns]
prob = model.predict_proba(scaler.transform(new_customer))[0][1]
print("\nNew customer: month-to-month, fiber, 5 months tenure, pays by electronic check")
print("Chance of leaving: {:.1f}%".format(prob * 100))
