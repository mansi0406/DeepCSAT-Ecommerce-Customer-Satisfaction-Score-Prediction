# DeepCSAT – E-Commerce Customer Satisfaction Prediction

## 📌 Project Overview

DeepCSAT is a **Machine Learning–based system designed to predict Customer Satisfaction (CSAT) scores for e-commerce customer support interactions**. The project analyzes structured ticket information along with textual customer remarks to determine whether the satisfaction level will be **high or low**.

The system helps support teams understand the key factors affecting customer satisfaction and provides predictive insights that can help improve service quality.

The project includes:

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning model training
* Model evaluation
* A Streamlit web application for real-time CSAT prediction

---

# 🎯 Objectives

The main objectives of this project are:

* Predict customer satisfaction scores using support ticket data
* Analyze factors affecting customer satisfaction
* Use customer remarks for sentiment-based insights
* Provide a simple interface for predicting CSAT using a web application

---

# 📊 Dataset Description

The dataset contains customer support interaction data from an e-commerce platform.

Each record represents a support ticket and contains information about the issue, the support agent, response time, and customer feedback.

### Key Features

| Feature           | Description                                |
| ----------------- | ------------------------------------------ |
| CSAT Score        | Customer satisfaction rating (1–5)         |
| response_time_hrs | Time taken by support agent to respond     |
| channel_name      | Communication channel (Phone, Chat, Email) |
| category          | Issue category                             |
| Sub-category      | Specific issue type                        |
| Agent Shift       | Agent working shift                        |
| Tenure Bucket     | Experience level of the support agent      |
| clean_remarks     | Customer textual feedback                  |

The dataset contains both **structured features** and **unstructured text data**.

---

# 🔎 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to identify patterns and relationships between different variables and customer satisfaction.

### Key Analyses Performed

#### 1. CSAT Distribution

* Most customers provided high ratings.
* Approximately **78% of customers rated CSAT as 5**, indicating strong positive feedback.

#### 2. Response Time vs CSAT

* Faster response times tend to result in higher satisfaction.
* Longer response times often correspond to lower CSAT scores.

#### 3. Hourly and Weekend Patterns

* Customer satisfaction varies based on time of interaction.
* Evening hours (6 PM – 10 PM) and weekends show slightly lower satisfaction levels.

#### 4. Agent & Shift Analysis

* Experienced agents tend to receive higher CSAT scores.
* Certain work shifts show slightly different satisfaction trends.

#### 5. Remarks-Based Insights

* Customer remarks were analyzed using sentiment analysis.
* Negative remarks strongly correlate with low CSAT scores.
* Positive remarks correspond to high satisfaction.

#### 6. Category-wise CSAT Analysis

* Some issue categories consistently receive lower satisfaction ratings.
* These categories may require operational improvements.

---

# ⚙️ Feature Engineering

Several new features were created to improve model performance.

### Created Features

**Sentiment Score**

Customer remarks were analyzed using sentiment analysis to determine emotional tone.

Range:

* -1 → Negative sentiment
* 0 → Neutral sentiment
* +1 → Positive sentiment

**Remarks Length**

The number of characters in the customer remark.

**Response Time**

Time taken by support agents to respond to the customer query.

These features help the model better understand customer experiences.

---

# 🤖 Machine Learning Models

The following machine learning algorithms were trained to predict CSAT:

### Logistic Regression

A simple and interpretable classification algorithm used as a baseline model.

### Random Forest

An ensemble learning model that builds multiple decision trees and combines their predictions.

Advantages:

* Handles complex relationships
* Reduces overfitting
* Good predictive performance

### XGBoost

A gradient boosting algorithm known for high accuracy and efficiency.

Advantages:

* Strong predictive power
* Handles structured data well
* Widely used in industry applications

---

# 📈 Model Evaluation

Models were evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score

The best-performing model was selected for deployment in the web application.

---

# 💻 Streamlit Web Application

A web interface was built using **Streamlit** to allow users to interact with the model.

The application enables users to input support ticket details and receive CSAT predictions instantly.

### Application Workflow

1. User enters customer remark.
2. User provides ticket details such as response time and issue category.
3. Sentiment analysis is performed on the remark.
4. The selected machine learning model predicts the CSAT score.
5. The prediction result and probability are displayed.

---

# 🌐 Website Interface

The Streamlit application contains the following sections.

### Input Section

Users provide:

* Customer remarks
* Agent shift
* Tenure bucket
* Communication channel
* Issue category
* Response time

### Prediction Section

After clicking **Predict**, the system displays:

* Predicted CSAT label
* Prediction probability

### Feature Importance

For tree-based models, the system also displays the **most influential features affecting the prediction**.

---

# 🏗 Project Structure

```
Deep-CSAT-Project
│
├── artifacts
│   ├── logreg_pipeline.joblib
│   ├── rf_pipeline.joblib
│   ├── xgb_pipeline.joblib
│   ├── label_encoder.joblib
│   └── categorical_values.json
│
├── notebooks
│   └── DeepCSAT_Ecommerce.ipynb
│
├── data
│   └── eCommerce_Customer_support_data.csv
│
├── main
│   └── app.py
│
└── README.md
```

---

# 🚀 How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn textblob xgboost joblib
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

### 3️⃣ Open the browser

```
http://localhost:8501
```

---

# 📌 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TextBlob
* Streamlit
* Matplotlib
* Seaborn

---

# 📊 Key Insights

* Response time negatively impacts customer satisfaction.
* Sentiment in customer remarks strongly correlates with CSAT scores.
* Experienced agents generally receive better satisfaction ratings.
* Certain issue categories consistently produce lower satisfaction scores.

---

# 🔮 Future Improvements

Possible improvements include:

* Using deep learning models for text analysis
* Integrating real-time support data
* Creating advanced dashboards for business analytics
* Adding explainable AI tools such as SHAP for model interpretation

---

# 📚 Conclusion

DeepCSAT demonstrates how machine learning and natural language processing can be used together to predict customer satisfaction. By analyzing support ticket data and customer feedback, the system provides valuable insights that can help companies improve their customer service operations.

---

# 👩‍💻 Author
MANSI SONI
Project developed as part of an academic machine learning project on **Customer Satisfaction Prediction for E-Commerce Support Systems**.
