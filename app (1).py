import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# عنوان التطبيق
st.title("🌟 Emergency Patient Prediction 🌟")
st.markdown("### **توقع أعداد المرضى بناءً على عوامل الطقس، المناسبات، والأمراض**")

# تحميل البيانات
data = pd.read_csv("emergency_data_2024_clean.csv")

# تجهيز البيانات
event_mapping = {value: idx for idx, value in enumerate(data["Event"].unique())}
weather_mapping = {value: idx for idx, value in enumerate(data["Weather Condition"].unique())}
disease_mapping = {value: idx for idx, value in enumerate(data["Disease Type"].unique())}

data["Event"] = data["Event"].map(event_mapping)
data["Weather Condition"] = data["Weather Condition"].map(weather_mapping)
data["Disease Type"] = data["Disease Type"].map(disease_mapping)

X = data[["Temperature (°C)", "Event", "Weather Condition", "Disease Type"]]
y = data["Patients"]

# تدريب نموذج Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# المدخلات من المستخدم
st.sidebar.header("إدخال البيانات")
date_input = st.sidebar.date_input("اختر التاريخ", datetime(2024, 1, 1))

# حساب اليوم تلقائيًا بناءً على التاريخ
day_of_year = date_input.timetuple().tm_yday  # اليوم في السنة (1-365)

# ضبط القيم بناءً على التاريخ المدخل
temperature = 25 + (day_of_year % 20 - 10)  # درجة الحرارة تتغير بناءً على اليوم
event = list(event_mapping.keys())[day_of_year % len(event_mapping)]  # اختيار المناسبة
weather = list(weather_mapping.keys())[day_of_year % len(weather_mapping)]  # حالة الطقس
disease = list(disease_mapping.keys())[day_of_year % len(disease_mapping)]  # نوع المرض

# عرض القيم المحسوبة
st.sidebar.markdown(f"### القيم المحسوبة بناءً على التاريخ:")
st.sidebar.write(f"- درجة الحرارة: {temperature}°C")
st.sidebar.write(f"- المناسبة: {event}")
st.sidebar.write(f"- حالة الطقس: {weather}")
st.sidebar.write(f"- نوع المرض: {disease}")

# تجهيز بيانات الإدخال
input_data = pd.DataFrame({
    "Temperature (°C)": [temperature],
    "Event": [event_mapping[event]],
    "Weather Condition": [weather_mapping[weather]],
    "Disease Type": [disease_mapping[disease]]
})

# التنبؤ باستخدام النموذج
prediction = model.predict(input_data)

# استخراج اسم اليوم من التاريخ
day_name = date_input.strftime('%A')

# عرض النتائج بشكل جذاب
st.markdown(f"""
### **📅 التاريخ:** {date_input.strftime('%Y-%m-%d')} ({day_name})
### **👨‍⚕️ عدد المرضى المتوقعين:** {int(prediction[0])} مريض
""")

# ملاحظة: تحسين النصوص
st.info("🔍 يعتمد هذا التوقع على البيانات التاريخية لعوامل الطقس والمناسبات والأمراض.")