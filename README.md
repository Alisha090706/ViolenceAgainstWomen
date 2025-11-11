# 🧠 Violence Against Women Prediction Using Machine Learning

## 📄 Overview
This project analyzes and predicts the justification of violence against women (VAW) based on demographic and socio-cultural factors such as **age**, **education**, **employment**, **marital status**, and **residence**.  
The goal is to uncover patterns that reflect how social norms and gender inequality influence attitudes toward violence and to develop a machine learning model capable of identifying high-risk factors.

---

## 🎯 Objectives
- To study how demographic variables affect the justification of violence.  
- To detect psychological and social patterns underlying acceptance of VAW.  
- To apply machine learning algorithms for predicting justification trends.  
- To raise awareness about how education, employment, and residence shape perceptions of gender-based violence.

---

## 🧩 Dataset
**Source:** Derived from large-scale global survey datasets (e.g., DHS, UN Data).

**Features:**
- Country  
- Gender  
- Demographics Question (e.g., Education, Age, Employment)  
- Demographics Response  
- Question (justification scenarios)  
- Survey Year  
- Value (percentage of justification)

Each record represents the percentage of respondents who justified violence under specific demographic and situational conditions.

---

## ⚙️ Methodology

### 🧹 Data Cleaning & Preprocessing
- Handled missing and null values  
- Removed irrelevant features (`RecordID`, `Survey Year`)  
- Standardized text and categorical labels  

### 📊 Exploratory Data Analysis (EDA)
- Visualized justification trends by demographic categories  
- Identified outliers and data skew  
- Observed normalization patterns (e.g., higher justification among women or rural respondents)

### ⚒️ Feature Engineering
- Converted categorical features into numerical format using **Label Encoding / One-Hot Encoding**  
- Scaled numerical variables  

### 🤖 Model Training
Implemented and compared multiple algorithms:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM)  

Models were evaluated based on **accuracy**, **precision**, **recall**, and **F1-score**.

### 📈 Model Evaluation & Interpretation
- Analyzed feature importance  
- Interpreted socio-demographic correlations  
- Visualized prediction outputs to highlight vulnerable groups

---

## 💡 Key Insights
- **Education** and **residence** are major determinants — justification decreases as education level rises.  
- **Rural women** show higher acceptance of violence than urban women.  
- **Females** sometimes justify violence more than males, revealing internalized patriarchal beliefs and social conditioning.  
- **Younger age groups (15–24)** show slightly higher normalization of violence compared to older groups.

---

## 🧰 Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** Git & GitHub

---

## 🚀 Future Scope
- Extend model with **deep learning** techniques for regional analysis  
- Integrate **Natural Language Processing (NLP)** for survey text interpretation  
- Build an **interactive dashboard** to visualize global patterns and awareness levels  

---

## 👩‍💻 Author
**Alisha Kapoor**  
*Computer Science & AI Engineering Student*  
📧 alishakapoor0907@gmail.com, alisha021btcseai24@igdtuw.ac.in 
**Bhawya Kumari**  
*Computer Science & AI Engineering Student*  
📧 bhawya048btcseai24@igdtuw.ac.in 

