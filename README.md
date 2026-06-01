# 42_Linear_Regression

# 📈 Linear Regression from Scratch (Gradient Descent)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

A **from-scratch implementation of univariate linear regression** using **batch gradient descent**, without any machine learning libraries.

The model predicts **car price based on mileage**.

---

## 🎯 Objective

Build a complete machine learning pipeline including:

- supervised learning
- gradient descent optimization
- feature normalization
- model persistence (JSON serialization)
- prediction system
- performance evaluation (MAE / R²)

---

## 🧠 Model


price = θ0 + θ1 × mileage


- θ0: bias (intercept)
- θ1: weight for mileage

---

## ⚙️ Pipeline Overview

### 1. Training
- load dataset (CSV)
- normalize features (mean / standard deviation)
- optimize parameters using batch gradient descent
- minimize cost function
- save trained model to JSON

### 2. Inference
- load model from JSON
- normalize user input
- compute prediction
- denormalize output

### 3. Evaluation
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

---

## 🔧 Techniques Used

- Linear Regression
- Batch Gradient Descent
- Feature Scaling (Z-score normalization)
- Mean Squared Error cost function
- Model serialization (JSON)

---

## 📐 Formulas

### Hypothesis function

h(x) = θ0 + θ1 × x


### Cost function

J(θ) = (1 / 2m) × Σ (h(xᵢ) - yᵢ)²


### Mean Absolute Error (MAE)

MAE = (1 / m) × Σ |yᵢ - ŷᵢ|


### R² Score

R² = 1 - (SS_res / SS_tot)

---

## 🚀 Usage

### Train model

```
python training.py
```

### Predict

```
python prediction.py
```

## Example:

```
Mileage : 75000
→ Predicted Price : 6800 €
```

### Evaluate

```
python precision.py
```

---

## 📌 Key Insights

- Feature scaling is essential for stable gradient descent  
- Learning rate strongly affects convergence speed and stability  
- MAE provides intuitive error interpretation (in euros)  
- R² measures how much variance is explained by the model  
- End-to-end ML pipeline implemented without external ML frameworks  

---

## 📈 What I Learned

- Fundamentals of gradient descent optimization  
- Impact of data scaling on convergence behavior  
- Evaluation metrics for regression models  
- Full machine learning pipeline design  
- Model serialization and inference workflow  
