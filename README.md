# housing-prediction-analysis.

# Fraud Detection in Housing Dataset

## Introduction
This project focuses on **fraud detection** in a housing dataset using **unsupervised anomaly detection** techniques. The objective is to identify unusual patterns or deviations that may indicate fraudulent activities.

## Dataset Overview
The dataset consists of **545 transactions** and **13 features**, including:
- **Numerical features:** `price`, `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- **Categorical features:** `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

The target variable (fraud indicator) was not explicitly provided, so **anomaly detection** was applied to identify potential fraudulent cases.

## Methodology
### 1. Data Preprocessing
- **Encoding categorical features:** Categorical variables were converted into numerical representations using **Label Encoding**.
- **Feature selection:** The `price` column was excluded from anomaly detection to avoid bias.

### 2. Anomaly Detection using Isolation Forest
We implemented **Isolation Forest**, a widely used unsupervised machine learning algorithm for anomaly detection.

#### **Model Parameters:**
- `n_estimators = 100` (number of trees in the forest)
- `contamination = 0.05` (assumed fraction of anomalies)
- `random_state = 42` (for reproducibility)

The model assigns an **anomaly score** to each transaction, identifying potential fraudulent activities.

## Results
### **Anomaly Detection Outcomes**
- **28 transactions** were detected as **anomalies (potential fraud cases)**.
- **517 transactions** were classified as **normal**.

### **Key Findings**
- **Outliers in price and area**: Some properties had significantly higher/lower prices and areas compared to the general trend.
- **High correlation with categorical features**: Transactions marked as anomalies often had unusual combinations of features such as **basements, guest rooms, or hot water heating**.
- **Potential fraud indicators**: Transactions with extreme values in `bathrooms`, `stories`, or `parking` counts were more likely to be anomalies.


## Repository Structure
```
|-- data/                 # Raw and processed datasets
|-- notebooks/            # Jupyter notebooks for analysis
|-- src/                  # Scripts for data processing and modeling
|-- results/              # Outputs and anomaly reports
|-- README.md             # Project overview and instructions
|-- requirements.txt      # Dependencies and packages
```

## Dependencies
This project uses the following Python libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Running the Code
To execute the anomaly detection analysis:
```python
python src/anomaly_detection.py
```

## Conclusion
This project provides a **baseline fraud detection system** using **unsupervised learning**. Future improvements include integrating **real-time fraud detection** and **predictive modeling** to enhance accuracy and scalability.

