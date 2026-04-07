# 🩺 Diabetes Prediction Web App

A Flask-based Machine Learning web application that predicts whether a person is diabetic or not using a Support Vector Machine (SVM) model. The app also provides data visualizations and model insights for better understanding.

---

##  Features

- 🔍 Predict diabetes using ML model  
- 📊 Display prediction confidence (probability)  
- 📈 Data visualizations (histogram, heatmap, scatter plots)  
- 🤖 Model accuracy and feature importance  
- 🌐 Simple and interactive web interface  

---

## 🛠️ Tech Stack

- Python  
- Flask  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## 📂 Project Structure

```
app.py
diabetes.csv
templates/
│ ├── index.html
│ ├── visualize.html
│ └── model.html
static/
│ └── plots/
```
---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/khanjamal59/diabetes_predict.git
cd diabetes_predict
pip install flask pandas numpy scikit-learn matplotlib seaborn
python app.py
