"""
dashboard.py
------------
Streamlit dashboard with EN/RU language support.
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SOC MRV Dashboard", layout="wide")

# Language Dropdown
lang = st.sidebar.selectbox("Language / Язык", ["English", "Русский"])

T = {
    "English": {
        "title": "Soil Organic Carbon — Decision Support",
        "upload": "Upload farm data (CSV)",
        "soc": "Predicted SOC (%)",
        "credits": "Estimated carbon credits (t CO2-eq)",
        "info": "Model not yet wired in. Coming after baseline validation."
    },
    "Русский": {
        "title": "Органический углерод почвы — поддержка решений",
        "upload": "Загрузите данные хозяйства (CSV)",
        "soc": "Прогноз ООУ (%)",
        "credits": "Оценка углеродных единиц (т CO₂-экв.)",
        "info": "Модель ещё не подключена. Будет добавлено после валидации."
    }
}

t = T[lang]

st.title(t["title"])
uploaded = st.file_uploader(t["upload"], type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    col1.metric(t["soc"], "—")
    col2.metric(t["credits"], "—")
    
    st.info(t["info"])
