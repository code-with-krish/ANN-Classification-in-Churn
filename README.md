# 📉 Customer Churn Classification using Artificial Neural Network (ANN)

This project predicts whether a customer is likely to **churn (leave the service)** using an **Artificial Neural Network (ANN)** built with TensorFlow/Keras.  
It helps telecom, banking, or service-based industries retain valuable customers by proactively identifying potential churners.

---

## 🚀 Features

- End-to-End ANN Model for binary classification
- Data preprocessing including label encoding and feature scaling
- Training, evaluation, and visualization of performance
- Web-based GUI app using **Streamlit**
- Confusion matrix, accuracy, and interactive insights

---

## 📊 Dataset Info

- **Source:** [Kaggle / Telco Customer Churn Dataset]
- **Features Include:**
  - Credit Score
  - Geography (France, Spain, Germany)
  - Gender, Age
  - Tenure, Balance
  - Number of Products
  - Has Credit Card, Is Active Member
  - Estimated Salary
  - Exited (Target Variable)

---

## 🧠 Model Architecture (ANN)

- **Input Layer:** Based on number of features after preprocessing
- **Hidden Layers:** 2 Dense layers with ReLU activation
- **Output Layer:** 1 neuron with Sigmoid activation
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn
- Streamlit

---

## 📦 Installation & Run

> 💡 Make sure Python 3.8+ and pip are installed before running.

```bash
# Clone the repository
git clone https://github.com/code-with-krish/ANN-Classification-in-Churn.git
cd ANN-Classification-in-Churn

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run streamlit_app.py
