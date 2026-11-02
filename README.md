# SL_Mini_project_Fraud_detection
Bank transaction fraud detection ML model
A machine learning mini-project exploring transaction data, preparing it for analysis, and training a Logistic Regression classifier to identify potentially fraudulent transactions.
Project Overview
This project performs exploratory data analysis (EDA), data cleaning, feature preprocessing, and binary classification on a transaction dataset. The target variable, is_fraud, indicates whether a transaction is fraudulent.

The notebook covers:
- Dataset inspection and descriptive statistics
- Missing-value analysis and imputation
- Outlier detection and IQR-based capping
- Min-max feature normalization
- Correlation analysis
- Logistic Regression model training
- Model evaluation using accuracy, confusion matrix, classification report, and ROC-AUC
Important evaluation note: The notebook's test accuracy is 89.36%, but the model predicts every test transaction as non-fraud. Fraud recall is 0.00%, so the current model is not useful for detecting fraud despite its accuracy.

Dataset
The notebook loads a local CSV file named fraud.csv.
- Rows: 7,000
- Columns: 13
- Target: is_fraud
- Missing values: Present in 12 input features; the target has no missing values.
Features
Feature	Description
transaction_amount	Transaction amount
hour_of_day	Hour when the transaction occurred
is_weekend	Weekend indicator
num_items	Number of items in the transaction
customer_age	Customer age
prev_transactions	Number of previous transactions
distance_from_home	Distance from the customer's home
device_type	Device category/code
network_quality	Network quality measure
is_first_transaction	Indicator for a customer's first transaction
store_type	Store category/code
velocity_score	Transaction velocity score
is_fraud	Target label: fraudulent or non-fraudulent

1. Load and inspect data
The notebook uses Pandas and NumPy to load and inspect the dataset, including its dimensions, data types, summary statistics, and sample rows.
2. Analyze missing values
Missing values are counted and visualized by feature.
3. Impute missing values
A copy of the dataset is created for preprocessing:
- Numerical features are filled with their median.
- Categorical/discrete features are filled with their mode.
After imputation, the notebook reports zero remaining missing values.
4. Detect and cap outliers
Box plots are used to inspect transaction amount and distance from home. The notebook then applies the Interquartile Range (IQR) rule to cap outliers in selected features:

transaction_amount, customer_age, prev_transactions, distance_from_home, network_quality, velocity_score, and num_items.

Values below Q1 - 1.5 × IQR or above Q3 + 1.5 × IQR are clipped to the corresponding boundary.
5. Normalize features
Min-max normalization scales each input feature to the range [0, 1]. The target column is_fraud is excluded from normalization.

A correlation heatmap is generated to inspect relationships between processed features.
6. Train the classifier
The data is split into training and test sets using an 80/20 split and random_state=42.
- Training samples: 5,600
- Test samples: 1,400
- Algorithm: Logistic Regression
7. Evaluate the model
The notebook evaluates predictions using:
- Accuracy
- Confusion matrix
- Precision, recall, and F1-score
- ROC curve and ROC-AUC
Results
Test-set performance
Metric	Result
Accuracy	89.36%
Fraud precision	0.00%
Fraud recall	0.00%
Fraud F1-score	0.00%
Test samples	1,400


Confusion matrix
Rows represent actual labels; columns represent predicted labels.
	Predicted: Not Fraud	Predicted: Fraud
Actual: Not Fraud	1,251	0
Actual: Fraud	149	0


The model correctly classifies all 1,251 non-fraud transactions but misses all 149 fraud cases. The dataset's class imbalance makes accuracy misleading as a standalone metric.

Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

Project Contributors
- Om Supe — A5-B2-19
- Shriraj Dhore — A5-B1-02
- Sahil Bhure — A5-B2-30
Academic mini-project focused on exploratory data analysis and a baseline fraud classification workflow.
