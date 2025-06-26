from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

data = pd.read_csv("diabetes.csv")
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

x_train, x_test, y_train, y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(x_train, y_train)

accuracy = accuracy_score(y_test, classifier.predict(x_test))
coefficients = classifier.coef_[0]
features = X.columns

os.makedirs("static/plots", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['preg']),
            float(request.form['glu']),
            float(request.form['bp']),
            float(request.form['skin']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age']),
        ]
        input_array = np.array(inputs).reshape(1, -1)
        std_input = scaler.transform(input_array)
        prediction = classifier.predict(std_input)[0]
        prob = classifier.predict_proba(std_input)[0][1]

        result = "Diabetic" if prediction == 1 else "Non Diabetic"
        return render_template("index.html", result=result, confidence=round(prob * 100, 2))
    except:
        return render_template("index.html", result="Error in input")

@app.route('/visualize')
def visualize():
    fig1, ax1 = plt.subplots()
    sns.histplot(data['Glucose'], bins=30, kde=True, ax=ax1, color='skyblue')
    ax1.set_title("Glucose Distribution")
    fig1.savefig('static/plots/glucose.png')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Correlation Heatmap")
    fig2.savefig('static/plots/heatmap.png')
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    sns.countplot(x='Outcome', data=data, palette='coolwarm', ax=ax3)
    ax3.set_title("Outcome Count")
    fig3.savefig('static/plots/outcome.png')
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=data, x='BMI', y='Age', hue='Outcome', style='Outcome', palette='coolwarm', s=100, ax=ax4)
    ax4.set_title("BMI vs Age by Outcome")
    fig4.savefig('static/plots/bmi_age.png')
    plt.close(fig4)

    return render_template("visualize.html")

@app.route('/model')
def model_info():
    fig5, ax5 = plt.subplots()
    sns.barplot(x=coefficients, y=features, palette='viridis', ax=ax5)
    ax5.set_title("Feature Importance (SVM)")
    fig5.savefig("static/plots/importance.png")
    plt.close(fig5)

    return render_template("model.html", accuracy=round(accuracy * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
