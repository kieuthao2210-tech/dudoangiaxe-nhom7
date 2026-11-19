import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Dự Đoán Giá Xe Ô Tô - Nhóm 7", page_icon="🚗", layout="wide")

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

def read_dataset(file_path):
    try:
        return pd.read_excel(file_path, engine='openpyxl')
    except Exception:
        for enc in ['utf-8', 'latin1', 'cp1252', 'utf-16', 'ISO-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    return None

@st.cache_resource
def load_and_train_model(max_depth_val):
    df_raw = read_dataset('train.csv')
    if df_raw is None or 'Price' not in df_raw.columns:
        st.error("⚠️ Không đọc được file train.csv!")
        st.stop()
        
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
    
    # Chia tập Train (80%) và Validation/Test (20%) từ train.csv để tính độ chính xác
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor(max_depth=max_depth_val, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict(X_val)
    r2 = r2_score(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    non_zero = y_val != 0
    mape = np.mean(np.abs((y_val[non_zero] - y_pred_val[non_zero]) / y_val[non_zero])) * 100
    
    metrics = {
        'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape,
        'train_size': len(X_train), 'val_size': len(X_val)
    }
    
    return model, label_encoders, raw_options, features, df_raw, metrics

# --- GIAO DIỆN CHÍNH ---
st.title("🚗 Hệ Thống Dự Đoán Giá Xe Ô Tô Cũ — Cây Quyết Định")
st.markdown("Thuật toán: **Decision Tree Regressor** | Thực hiện: **Nhóm 7**")

st.sidebar.header("🌳 Cấu hình Cây quyết định")
max_depth = st.sidebar.slider("Độ sâu tối đa của cây (max_depth)", min_value=3, max_value=25, value=10)

model, label_encoders, raw_options, features, df_raw, metrics = load_and_train_model(max_depth)

# --- PHẦN 1: CHỈ SỐ ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---
st.subheader("📊 1. Các chỉ số đánh giá độ chính xác của mô hình (Tập kiểm thử 20%)")
st.caption(f"Dữ liệu huấn luyện: **{metrics['train_size']:,} xe** | Dữ liệu kiểm thử đánh giá: **{metrics['val_size']:,} xe**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("R² Score (Độ chính xác)", f"{metrics['R2']:.4f}", f"{metrics['R2']*100:.2f}%")
m2.metric("MAE (Sai số tuyệt đối TB)", f"${metrics['MAE']:,.2f}")
m3.metric("RMSE (Sai số bình phương TB)", f"${metrics['RMSE']:,.2f}")
m4.metric("MAPE (Sai số phần trăm TB)", f"{metrics['MAPE']:.2f}%")

st.markdown("---")

# --- PHẦN 2: DỰ ĐOÁN CHO 1 CHIẾC XE ---
st.subheader("🔍 2. Nhập thông số xe để dự đoán giá")

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

if st.button("🚀 DỰ ĐOÁN GIÁ XE NGAY", type="primary", use_container_width=True):
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
        input_df[col] = le.transform([val])[0] if val in le.classes_ else 0
            
    input_df = input_df[features]
    predicted_price = float(model.predict(input_df)[0])
    
    st.success("✅ Đã tính toán xong mức giá ước lượng!")
    st.metric(label="Giá xe dự đoán (Predicted Price)", value=f"${predicted_price:,.2f} USD")

st.markdown("---")

# --- PHẦN 3: TỰ ĐỘNG HIỂN THỊ KẾT QUẢ TRÊN FILE test.csv CÓ SẴN ---
st.subheader("📂 3. Kết quả dự đoán trên tập dữ liệu Test (file test.csv)")

df_test = read_dataset('test.csv')
if df_test is None:
    st.warning("⚠️ Chưa tìm thấy file `test.csv` trên GitHub. Hãy upload file `test.csv` vào cùng thư mục với `app.py` để xem kết quả tự động.")
else:
    df_test_clean = clean_data(df_test)
    X_test_file = df_test_clean[features].copy()
    for col, le in label_encoders.items():
        X_test_file[col] = X_test_file[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    preds_test = model.predict(X_test_file)
    result_df = df_test.copy()
    result_df.insert(0, 'Giá dự đoán ($)', np.round(preds_test, 2))
    
    # Nếu file test.csv có cột Price thì tính luôn chỉ số độ chính xác
    if 'Price' in result_df.columns:
        y_true_t = pd.to_numeric(result_df['Price'], errors='coerce')
        valid = y_true_t.notna() & (y_true_t > 0)
        if valid.sum() > 0:
            r2_t = r2_score(y_true_t[valid], preds_test[valid])
            mae_t = mean_absolute_error(y_true_t[valid], preds_test[valid])
            rmse_t = np.sqrt(mean_squared_error(y_true_t[valid], preds_test[valid]))
            mape_t = np.mean(np.abs((y_true_t[valid] - preds_test[valid]) / y_true_t[valid])) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score (File Test)", f"{r2_t:.4f}")
            c2.metric("MAE (File Test)", f"${mae_t:,.2f}")
            c3.metric("RMSE (File Test)", f"${rmse_t:,.2f}")
            c4.metric("MAPE (File Test)", f"{mape_t:.2f}%")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng số xe tập Test", f"{len(result_df):,} xe")
        c2.metric("Giá dự đoán trung bình", f"${np.mean(preds_test):,.2f}")
        c3.metric("Giá dự đoán thấp nhất", f"${np.min(preds_test):,.2f}")
        c4.metric("Giá dự đoán cao nhất", f"${np.max(preds_test):,.2f}")
        
    st.dataframe(result_df.head(50), use_container_width=True)
    csv_out = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="⬇️ Tải toàn bộ file kết quả dự đoán tập Test (.CSV)",
        data=csv_out,
        file_name="ket_qua_du_doan_test.csv",
        mime="text/csv"
    )
