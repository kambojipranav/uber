# 🚖 Uber Trip Forecasting App using XGBoost + Streamlit

[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?logo=python)](https://xgboost.readthedocs.io/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

> ✨ An intelligent time series forecasting dashboard that predicts hourly Uber trips using advanced machine learning.

---

## 🌐 Live Demo

🧪 **Try the app here** 👉 [Streamlit Live App](https://kuangedathvnaesfg4eazn.streamlit.app/)

---

## 📈 About the Project

This project is a real-world ML dashboard where we:

- Forecast hourly Uber ride counts using **lag features**.
- Train a custom **XGBoost regression model**.
- Visualize performance via **Plotly** charts.
- Offer a slick **dark/light theme toggle**.
- Host the entire experience via **Streamlit Cloud**.

It brings together real-time forecasting, clean UI/UX, and fast machine learning — all in one smooth app.

---

## 🧠 Key Features

- ✅ **Time Series Forecasting** using Lag Features  
- 📊 **Dynamic Visualization** of Actual vs Predicted  
- 🌙 **Light/Dark Mode** with modern gradients  
- ⚙️ **Sidebar Controls** (Lag Window, Forecast Trigger)  
- 📈 **MAPE & R²** Accuracy Scores  
- 📦 **Deployed on Streamlit Cloud**

---

## 📂 Project Structure



![image](https://github.com/user-attachments/assets/05b11ab1-f95e-4de1-a29b-c8dde64da5ba)


---

## 📊 Dataset

📁 **Source**: NYC Taxi & Limousine Commission (FOIL Request)  
🗓️ **Time Range**: Jan–Feb 2015  
📌 **Columns**:  
- `dispatching_base_number`  
- `date`  
- `active_vehicles`  
- `trips`

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/kambojipranav/uber.git
cd uber

# 2. Create virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

| Tool            | Description             |
| --------------- | ----------------------- |
| 🐍 Python       | Programming Language    |
| 🧠 XGBoost      | ML model for regression |
| 📊 Plotly       | Interactive plots       |
| 🌐 Streamlit    | Web UI framework        |
| 📁 Pandas       | Data wrangling          |
| ⚙️ Scikit-learn | Metrics (MAPE, R²)      |



