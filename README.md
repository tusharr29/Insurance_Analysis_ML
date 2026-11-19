# 🏥 Insurance Cost Prediction System

An end-to-end Machine Learning project that predicts individual medical insurance charges using Linear Regression. The application takes customer information such as age, gender, BMI, number of children, and smoking habits to estimate insurance costs.

🌐 **Live Application:** [Insurance Cost Prediction](https://insurance-analysisml-liinear-regression.streamlit.app/)

💻 **GitHub Repository:** [Insurance_Analysis_ML](https://github.com/tusharr29/Insurance_Analysis_ML)

---

## 📌 Project Overview

Medical insurance costs vary depending on several demographic and lifestyle factors. This project uses Machine Learning to identify relationships between these factors and insurance charges.

A Linear Regression model is trained on historical insurance data and deployed through an interactive Streamlit web application.

## 🎯 Project Objectives

- Analyse factors affecting medical insurance charges.
- Perform Exploratory Data Analysis (EDA).
- Apply data cleaning and preprocessing techniques.
- Perform categorical encoding and feature engineering.
- Analyse feature relationships using Pearson correlation.
- Apply Chi-square statistical testing for categorical feature analysis.
- Train a Linear Regression model.
- Evaluate model performance.
- Deploy an interactive prediction application.

## 📊 Dataset

**Dataset:** Medical Cost Personal Dataset (Insurance Dataset)

The dataset contains customer information and their corresponding medical insurance charges.

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender of the individual |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges (target) |

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Streamlit
- Joblib

## ⚙️ Machine Learning Workflow

### 1. Exploratory Data Analysis
- Dataset exploration and statistical summaries.
- Missing-value and duplicate analysis.
- Distribution analysis using histograms.
- Outlier detection using boxplots.
- Correlation analysis using heatmaps.

### 2. Data Cleaning and Preprocessing
- Duplicate removal.
- Categorical variable encoding.
- Feature engineering.
- StandardScaler for numerical features.

### 3. Statistical Analysis
- Pearson correlation analysis.
- Chi-square statistical testing.
- Feature relationship analysis.

### 4. Model Development
- Algorithm: Linear Regression.
- Train-test split: 80:20.
- Random state: 42.
- Model trained using five selected features:
  - Age
  - Gender
  - BMI
  - Number of children
  - Smoking status

### 5. Model Evaluation

The model is evaluated using:

- R-squared (R²)
- Adjusted R-squared
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## 🚀 Streamlit Application

The deployed application allows users to:

1. Enter their age.
2. Select gender.
3. Enter BMI.
4. Specify number of children.
5. Select smoking status.
6. Generate an estimated medical insurance cost.

The application uses the trained Linear Regression model and saved StandardScaler to generate predictions.

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/tusharr29/Insurance_Analysis_ML.git
```

Navigate to the project directory:

```bash
cd Insurance_Analysis_ML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python main.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Project Structure

```text
Insurance_Analysis_ML/
│
├── insurance.csv
├── main.py
├── app.py
├── insurance_model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔮 Future Improvements

- Compare Linear Regression with other regression algorithms.
- Implement cross-validation.
- Perform hyperparameter tuning.
- Improve model performance through feature selection.
- Add interactive data visualizations.
- Explore advanced regression techniques.

## 👨‍💻 Author

**Tushar Deshmukh**

- GitHub: [@tusharr29](https://github.com/tusharr29)
- Project: [Insurance Cost Prediction](https://insurance-analysisml-liinear-regression.streamlit.app/)

---

*This project was developed for educational and portfolio purposes. Predictions are estimates based on historical data and should not be treated as actual insurance quotes.*
