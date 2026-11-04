# Customer Churn Analysis

A beginner-level project that studies why customers leave a service (churn) and
predicts which customers may leave.

## What the project does
1. Loads the customer dataset (`data/customer_data.csv`)
2. Cleans it: removes duplicates, fixes the `total_charges` column (blank text values), fills missing numbers with the median, converts churn from Yes/No to 1/0
3. Explores churn across customer groups (contract type, internet service, payment method, tenure, monthly charges) with charts
4. Converts text columns to numbers (one-hot encoding) and scales the features
5. Trains a Logistic Regression model (80% train / 20% test)
6. Evaluates it with accuracy, a confusion matrix and a classification report

## Tools used
Python, Pandas, NumPy, Matplotlib, Scikit-learn

## How to run
```
pip install -r requirements.txt
python customer_churn.py
```
Charts are saved in the `outputs/` folder.

## Key findings (from this dataset)
- Overall churn rate is about 23%
- Month-to-month customers leave far more often (about 34%) than one-year (about 10%) or two-year (about 7%) customers
- Fiber optic customers churn more than DSL customers
- Customers who left had a shorter tenure and higher monthly charges on average

## Model results (test data)
| Metric | Value |
|--------|-------|
| Accuracy | 0.72 |
| Recall for customers who left | 0.57 |
| Precision for customers who left | 0.42 |

Only about 1 in 4 customers leave, so without balancing, the model reached about 78% accuracy
but found only 15% of the customers who actually left. I used `class_weight="balanced"`,
which lowers accuracy a little but finds many more of the customers who leave.

## Notes
- The dataset in this repo is a sample dataset of 1000 customers. To use real data
  (for example the Telco Customer Churn dataset), replace `data/customer_data.csv` with a
  file that has the same column names.

## Possible improvements
- Try Random Forest or Decision Tree and compare with Logistic Regression
- Use cross-validation
- Build a small Flask page to enter a customer's details and see the churn chance
