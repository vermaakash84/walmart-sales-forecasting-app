# 📊 Walmart Sales Forecasting App

This is a **Streamlit web application** that forecasts **weekly sales** for Walmart stores and departments using a trained **XGBoost model**.  
It supports **multi-step forecasting** (e.g., next 12 weeks) and lets users upload their own sales history to generate predictions.

---

## 🚀 Features
- Forecast future **weekly sales** by Store and Department.
- Supports **multi-step forecasts** (e.g., 4–52 weeks ahead).
- Interactive **line charts** for visualization.
- **Downloadable forecast CSV**.
- Built with:
  - 🐍 Python  
  - ⚡ XGBoost  
  - 📊 Streamlit  

---

## 📂 Repository Structure
```
walmart-sales-forecast-app/
│── app.py                     # Main Streamlit app
│── requirements.txt           # Dependencies
│── walmart_forecast_xgb.json  # Trained XGBoost model
│── README.md                  # Project documentation
│── LICENSE                    # Open-source license (MIT)
```

---

## 📦 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/walmart-sales-forecast-app.git
cd walmart-sales-forecast-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will start locally at:  
👉 [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deployment on Streamlit Cloud
1. Push this repo to GitHub.  
2. Go to [Streamlit Cloud](https://share.streamlit.io/).  
3. Select your repo and set the entrypoint as `app.py`.  
4. Deploy 🎉  

---

## 📊 Dataset
The app is based on the **[Walmart Sales Forecasting Dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)**.  
Please download it from Kaggle if you want to retrain the model.

---

## 📜 License
This project is licensed under the **MIT License**.  
Feel free to use, modify, and share!

---

## ✨ Author
Developed by **[Your Name]**  
🔗 Connect with me on [LinkedIn](https://linkedin.com)
