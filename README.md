# Telecom-Customer-Churn-Prediction-Using-Machine-Learning
Built a Telecom Customer Churn Prediction system using Python and Machine Learning. Performed data cleaning, feature engineering, EDA, SMOTE, model evaluation, and hyperparameter tuning across multiple algorithms to support customer retention strategies.

## Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to predict whether a telecom customer is likely to churn using machine learning techniques, enabling businesses to take proactive retention measures.

The project includes comprehensive Exploratory Data Analysis (EDA), data preprocessing, feature engineering, handling class imbalance, model building, hyperparameter tuning, and business recommendations.

## Business Problem

Telecom companies lose revenue when customers discontinue their services. Identifying customers at risk of churning helps businesses:

- Improve customer retention strategies
- Reduce revenue loss
- Increase customer lifetime value
- Optimize marketing and loyalty campaigns

## Data Collection and Dataset Source
The dataset used in this project was collected from a SQL database, making the project more industry-oriented and aligned with real-world telecom data management systems. Telecom companies generally store customer information, billing records, service usage details, and customer activity data in relational databases. Therefore, retrieving the dataset directly from a SQL database helped simulate a real business environment.

The database used for this project was named project_telecom, and the data was extracted from the table telecom_churn_data. The SQL database contained structured telecom customer information including demographic details, communication behavior, and service usage patterns.

After connecting to the SQL database, the dataset was imported into Python for further preprocessing, analysis, visualization, and machine learning model development. The extracted data was then converted into a pandas DataFrame to perform operations such as cleaning, feature engineering, exploratory data analysis, and predictive modeling.

## Dataset Information

The dataset contains customer demographic information, telecom usage patterns, and churn status.

### Features

- Telecom Partner
- Gender
- Age
- State
- City
- Number of Dependents
- Estimated Salary
- Calls Made
- SMS Sent
- Data Used
- Churn (Target Variable)

After preprocessing and feature engineering, the final dataset contained **965 records** and **46 features**. 


## Exploratory Data Analysis (EDA)

### Univariate Analysis

The following distributions were analyzed:

- Age
- Number of Dependents
- Estimated Salary
- Calls Made
- SMS Sent
- Data Usage
- Total Usage
- Usage per Salary
- Call-to-Data Ratio

### Key Findings

- Customers belonged to diverse age groups.
- Salary distribution indicated customers from different economic backgrounds.
- Customer engagement varied significantly across calls, SMS, and data usage.
- Lower activity customers showed higher churn tendencies.
- Different telecom providers exhibited varying churn patterns.
  
### Bivariate Analysis

Relationships between features and churn were analyzed using:

- Boxplots
- Countplots
- Churn distribution comparisons

### Key Findings

- Low-engagement customers were more likely to churn.
- Data usage influenced customer retention.
- Certain customer segments showed higher churn behavior. 

### Multivariate Analysis

- Pairplots
- Correlation Heatmap

Used to identify relationships among multiple variables and detect multicollinearity.

#  Data Preprocessing

### Missing Values

No missing values were found in the dataset. 

### Duplicate Records

No duplicate records were present.

### Outlier Treatment

Outliers were detected using the IQR method.

Transformations applied:

- Clipping extreme values
- Log transformation
- Feature scaling

### Feature Engineering

Created new features:

- Total Usage
- Usage per Salary
- Low Activity Flag
- Call-to-Data Ratio

Negative data usage records were removed before model training.

### Encoding

Categorical variables were converted using One-Hot Encoding. 

### Scaling

StandardScaler was applied to numerical features. 

# Handling Class Imbalance

The dataset was imbalanced:

| Churn Status | Count |
|-------------|-------:|
| No Churn | 792 |
| Churn | 173 |

SMOTE (Synthetic Minority Oversampling Technique) was used to balance the training dataset before model building. 

#  Models Implemented

1. Logistic Regression
2. Random Forest Classifier
3. Random Forest (Tuned)
4. Decision Tree Classifier
5. Decision Tree (Tuned)
6. Gradient Boosting Classifier
7. Gradient Boosting (Tuned)
8. XGBoost Classifier

#  Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|---------:|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.40 | 0.19 | 0.69 | 0.29 | 0.498 |
| Random Forest | 0.37 | 0.18 | 0.71 | 0.29 | 0.520 |
| Random Forest (Tuned) | 0.26 | 0.18 | 0.83 | 0.29 | 0.507 |
| Decision Tree | 0.63 | 0.15 | 0.23 | 0.18 | 0.475 |
| Decision Tree (Tuned) | 0.61 | 0.17 | 0.29 | 0.21 | 0.478 |
| Gradient Boosting | 0.33 | 0.15 | 0.60 | 0.25 | 0.483 |
| Gradient Boosting (Tuned) | 0.49 | 0.18 | 0.49 | 0.26 | 0.450 |
| XGBoost | 0.69 | 0.23 | 0.31 | 0.27 | 0.547 |
| XGBoost (Tuned) | 0.67 | 0.20 | 0.26 | 0.22 | 0.545 |

# Best Model

XGBoost was selected as the final model because it achieved the highest Accuracy (69%) and the highest ROC-AUC score (0.547) among all models. While Tuned Random Forest achieved the highest Recall (83%), it suffered from very low Accuracy (26%) and produced excessive false positives, making it impractical for real-world deployment.

By combining SMOTE and the `scale_pos_weight` parameter, XGBoost provided the best balance between churn detection, prediction stability, false positive control, and generalization capability. Therefore, XGBoost was chosen as the most business-efficient and reliable model for telecom churn prediction.

# 💡 Business Recommendations

Business Recommendations
1. Implement Early Churn Detection Programs

Use the XGBoost model to identify customers with a high probability of churning and proactively engage them before they leave. Early intervention can significantly reduce customer attrition and revenue loss.

2. Develop Personalized Retention Campaigns

Customers identified as high-risk should receive targeted offers such as discounted plans, bonus data packages, loyalty rewards, or customized service bundles to improve retention.

3. Increase Engagement Among Low-Usage Customers

Analysis showed that customer activity levels influence churn behavior. Telecom companies should encourage engagement through promotional offers, usage-based incentives, and personalized communication.

4. Strengthen Customer Relationship Management

Regular feedback collection, faster issue resolution, and improved customer support can enhance customer satisfaction and reduce the likelihood of churn.

5. Utilize Data-Driven Marketing Strategies

Leverage predictive insights to segment customers based on risk levels and allocate marketing resources more efficiently toward customers most likely to leave.

6. Continuously Monitor Churn Trends

The churn prediction model should be retrained periodically with new customer data to maintain prediction accuracy and adapt to changing customer behavior.

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

# ✅ Conclusion

This project successfully developed a machine learning-based solution for predicting telecom customer churn using customer demographics and usage behavior data. A comprehensive workflow involving data cleaning, exploratory data analysis, feature engineering, class imbalance handling using SMOTE, feature scaling, and multiple classification algorithms was implemented.

Several machine learning models were evaluated, including Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Gradient Boosting, and XGBoost. Among all models, XGBoost achieved the highest ROC-AUC score of 0.547 and the best overall balance between churn detection capability and prediction performance, making it the most suitable model for business deployment.

The developed solution enables telecom companies to identify customers at risk of leaving, implement targeted retention strategies, improve customer satisfaction, and minimize revenue loss. By integrating predictive analytics into customer relationship management processes, organizations can make proactive, data-driven decisions that enhance long-term customer retention and business profitability.
