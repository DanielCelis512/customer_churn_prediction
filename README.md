# Customer Churn Prediction
## Project Overview
This project explores customer churn in a telecommunications company and develops machine learning models to identify customers who are more likely to leave. The workflow covers data preparation, exploratory data analysis, model training, cross-validation, evaluation, and classification-threshold analysis.

The project was developed as an end-to-end data science portfolio project using Python and scikit-learn.
Business Problem
Customer churn can affect recurring revenue and increase customer acquisition costs. Identifying customers who may be at risk of leaving can help a business investigate potential issues and design customer-retention strategies.

The objective of this project is to analyze historical customer information and build a binary classification model that distinguishes between customers who churned and those who remained.
Important scope note: The available dataset contains historical customer and contract information. Some variables, such as cumulative charges and tenure derived from the dataset's reference date, may reflect information accumulated up to or after a customer's cancellation. Therefore, the results should be interpreted as retrospective classification performance on this dataset, not as proof that the model can predict future churn using only information available before a customer leaves. A production-oriented predictive system would require a clearly defined observation date and future prediction horizon.

## Dataset
The project uses four related CSV files:
- contract.csv: contract details, billing information, and customer start/end dates.
- personal.csv: demographic and household information.
- internet.csv: internet services subscribed to by each customer.
- phone.csv: phone-service information.
The tables are joined using customerID. Customers without a corresponding internet or phone record are retained, and the missing service entries are interpreted as not having that service.

The target variable, churn, is derived from EndDate:
- 1: the customer has a recorded end date.
- 0: the customer is still active (EndDate is recorded as No).
The dataset contains 7,043 customers. The target distribution is approximately 73.5% active customers and 26.5% churned customers.

The source data files are not included in this repository. Add them locally under data/final_provider/ if you have permission to use and redistribute them. Check the dataset provider's terms before publishing the raw data.
Project Workflow
## Data preparation
- Load and inspect the four source tables.
- Check customer identifiers and table relationships.
- Convert date and monetary fields to appropriate data types.
- Investigate missing values and duplicate records.
- Merge the tables into a customer-level dataset.
- Create the binary churn target and engineer tenure-related information.
- Remove identifiers and outcome-defining fields from the model features to avoid direct target leakage.

## Exploratory Data Analysis
- Examine the target distribution.
- Analyze numerical and categorical variables.
- Compare churn rates across customer groups.
- Explore numerical relationships and correlations.
- Summarize patterns that may help explain customer churn.

## Model development
The data is split into stratified training and test sets. The test set is held out for final evaluation.

The following classifiers are compared:
- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
Preprocessing is incorporated into scikit-learn pipelines and fitted within cross-validation. Numerical features are imputed, and categorical features are imputed and one-hot encoded. Scaling is applied to the Logistic Regression pipeline.

Model selection uses 5-fold Stratified Cross-Validation and ROC-AUC as the primary optimization metric. Additional metrics are tracked:
- Recall
- Precision
- F1-score
- Accuracy
No synthetic oversampling is used. Class-weight options are evaluated as part of model tuning.

## Final evaluation
The selected model is evaluated on the untouched test set. The notebook reports ROC-AUC, recall, precision, F1-score, accuracy, a classification report, a confusion matrix, and an ROC curve.

## Feature importance
Permutation importance is used to estimate how much each feature contributes to the model's ROC-AUC on the test set. These values describe predictive contribution in this evaluation; they do not establish causality.

## Classification threshold analysis
The default classification threshold of 0.50 is compared with a threshold selected using out-of-fold predictions from the training data. The selected threshold maximizes F1-score over the thresholds evaluated in the notebook.

This analysis illustrates the trade-off between identifying more churned customers (recall) and avoiding false alarms (precision).
Results
LightGBM achieved the strongest cross-validation ROC-AUC among the evaluated models.
Cross-validation and test performance
Metric	Cross-validation	Test set
ROC-AUC	0.9356	0.9348

Test-set metrics by classification threshold
Metric	Threshold 0.50	Threshold 0.40
Recall	0.7219	0.7701
Precision	0.8738	0.8205
F1-score	0.7906	0.7945
Accuracy	0.8985	0.8943
ROC-AUC	0.9348	0.9348

At a threshold of 0.40, the model identifies more churned customers: recall increases from 72.19% to 77.01%. This comes with lower precision, which means more customers who would not churn are flagged. The F1-score increases slightly, while accuracy decreases slightly.

The threshold should be selected according to the business cost of missed churners versus unnecessary retention interventions. A threshold of 0.40 is a reasonable recall-oriented alternative in this analysis, but it is not universally optimal.
Most influential features
The leading features according to permutation importance were:
1. TotalCharges
2. MonthlyCharges
3. tenure_days
4. Type
5. InternetService
Permutation importance is dataset- and model-dependent. It should not be interpreted as evidence that these variables cause churn.

## Technologies
- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- LightGBM
- Jupyter Notebook

## Limitations and Future Improvements
- Prospective prediction design: Define a consistent customer observation date and a future churn horizon. Build features only from information available at that date.
- Temporal validation: Evaluate the model on a later time period to better reflect deployment conditions.
- Business-oriented thresholding: Select a threshold using estimated retention costs, intervention capacity, and the financial impact of missed churners and false positives.
- Model monitoring: Track changes in customer behavior, data quality, and model performance over time.
- Interpretability: Complement permutation importance with methods such as SHAP, while distinguishing explanations from causal conclusions.

## Author
José Daniel Celis Lozano
- GitHub: DanielCelis512
- LinkedIn: jose-daniel-celis-lozano
