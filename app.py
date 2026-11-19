import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Du Doan Gia Xe O To - Nhom 7", page_icon="🚗", layout="wide")

def clean_data(df):
    df_clean = df.copy()
    df_clean['Mileage'] = df_clean['Mileage'].astype(str).str.replace('km', '', case=False).str.replace(' ', '')
    df_clean['Mileage'] = pd.to_numeric(df_clean['Mileage'], errors='coerce').fillna(0)
    
    df_clean['Levy'] = df_clean['Levy'].astype(str).str.replace('-', '')
    df_clean['Levy'] = pd.to_numeric(df_clean['Levy'], errors='coerce')
    df_clean['Levy'] = df_clean['Levy'].fillna(df_clean['Levy'].median())
    
    df_clean['Engine volume'] = df_clean['Engine volume'].astype(str).str.replace('Turbo', '', case=False).str.strip()
    df_clean['Engine volume'] = pd.to_numeric(df_clean['Engine volume'], errors='coerce').fillna(0.0)
    
    cols_to_drop = [col for col in ['ID', 'Doors'] if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    return df_clean

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

@st.cache_resource
def load_and_train_model():
    # Thu doc file voi nhieu bang ma khac nhau de tranh loi UnicodeDecodeError
    for enc in ['utf-8', 'latin1', 'cp1252', 'utf-16']:
        try:
            df_raw = pd.read_csv('train.csv', encoding=enc)
            break
        except Exception:
            continue
            
    df_clean = clean_data(df_raw)
    df_clean = remove_outliers_iqr(df_clean, 'Price')
    df_clean = remove_outliers_iqr(df_clean, 'Mileage')
    
    raw_options = {}
    categorical_cols = [
        'Manufacturer', 'Model', 'Category', 'Leather interior', 
        'Fuel type', 'Gear box type', 'Drive wheels', 'Wheel', 'Color'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str)
        raw_options[col] = sorted(df_clean[col].unique().tolist())
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
        
    features = [
        'Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 
        'Leather interior', 'Fuel type', 'Engine volume', 'Mileage', 
        'Cylinders', 'Gear box type', 'Drive wheels', 'Wheel', 'Color', 'Airbags'
    ]
    
    X = df_clean[features]
    y = df_clean['Price']
    
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X, y)
    return model, label_encoders, raw_options, features, df_raw

st.title("🚗 Hệ Thống AI Dự Đoán Giá Xe Ô Tô Cũ")
st.markdown("Mô hình **Decision Tree Regressor** — Thực hiện bởi **Nhóm 7**")

model, label_encoders, raw_options, features, df_raw = load_and_train_model()

col1, col2, col3 = st.columns(3)

with col1:
    manufacturer = st.selectbox("Hãng sản xuất (Manufacturer)", raw_options['Manufacturer'])
    available_models = sorted(df_raw[df_raw['Manufacturer'] == manufacturer]['Model'].astype(str).unique().tolist())
    model_car = st.selectbox("Dòng xe (Model)", available_models if available_models else raw_options['Model'])
    category = st.selectbox("Kiểu dáng (Category)", raw_options['Category'])
    prod_year = st.number_input("Năm sản xuất (Prod. year)", min_value=1950, max_value=2026, value=2016)
    color = st.selectbox("Màu sắc (Color)", raw_options['Color'])

with col2:
    fuel_type = st.selectbox("Loại nhiên liệu (Fuel type)", raw_options['Fuel type'])
    engine_volume = st.number_input("Dung tích động cơ (L)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    cylinders = st.number_input("Số xi-lanh (Cylinders)", min_value=1, max_value=16, value=4)
    gear_box = st.selectbox("Hộp số (Gear box type)", raw_options['Gear box type'])
    drive_wheels = st.selectbox("Hệ dẫn động (Drive wheels)", raw_options['Drive wheels'])

with col3:
    mileage = st.number_input("Số km đã đi (Mileage)", min_value=0, max_value=1000000, value=50000, step=1000)
    levy = st.number_input("Thuế (Levy)", min_value=0, max_value=15000, value=781)
    leather = st.selectbox("Nội thất da (Leather interior)", raw_options['Leather interior'])
    wheel = st.selectbox("Vị trí vô lăng (Wheel)", raw_options['Wheel'])
    airbags = st.slider("Số túi khí (Airbags)", min_value=0, max_value=16, value=12)

st.markdown("---")

if st.button("🔍 DỰ ĐOÁN GIÁ XE NGAY", type="primary", use_container_width=True):
    input_data = {
        'Levy': levy, 'Manufacturer': manufacturer, 'Model': model_car,
        'Prod. year': prod_year, 'Category': category, 'Leather interior': leather,
        'Fuel type': fuel_type, 'Engine volume': engine_volume, 'Mileage': mileage,
        'Cylinders': cylinders, 'Gear box type': gear_box, 'Drive wheels': drive_wheels,
        'Wheel': wheel, 'Color': color, 'Airbags': airbags
    }
    input_df = pd.DataFrame([input_data])
    
    for col, le in label_encoders.items():
        val = str(input_df[col].iloc[0])
        input_df[col] = le.transform([val]) if val in le.classes_ else 0
            
    input_df = input_df[features]
    predicted_price = model.predict(input_df)[0]
    
    st.success("✅ Đã tính toán xong mức giá ước lượng!")
    st.metric(label="Giá xe dự đoán (Predicted Price)", value=f"${predicted_price:,.2f} USD")
