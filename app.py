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

def read_dataset(file_source):
    try:
        return pd.read_excel(file_source, engine='openpyxl')
    except Exception:
        for enc in ['utf-8', 'latin1', 'cp1252', 'utf-16', 'ISO-8859-1']:
            try:
                if hasattr(file_source, 'seek'):
                    file_source.seek(0)
                df = pd.read_csv(file_source, encoding=enc, on_bad_lines='skip')
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    return None

@st.cache_resource
def load_and_train_model(max_depth_val):
    df_raw = read_dataset('train.csv')
    if df_raw is None or 'Price' not in df_raw.columns:
        st.error("⚠️ Không đọc được dữ liệu từ file train.csv. Hãy kiểm tra lại định dạng file!")
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
    
    # Chia tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Huấn luyện mô hình Cây quyết định (Decision Tree Regressor)
    model = DecisionTreeRegressor(max_depth=max_depth_val, random_state=42)
    model.fit(X_train, y_train)
    
    # Đánh giá trên tập Test
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Tính MAPE (tránh chia cho 0)
    non_zero_mask = y_test != 0
    mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / y_test[non_zero_mask])) * 100
    
    metrics = {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # Bảng so sánh thực tế vs dự đoán trên tập Test
    test_comparison = pd.DataFrame({
        'Giá thực tế (Actual $)': y_test.values,
        'Giá dự đoán (Predicted $)': np.round(y_pred_test, 2),
        'Chênh lệch ($)': np.round(np.abs(y_test.values - y_pred_test), 2)
    }).head(20)
    
    return model, label_encoders, raw_options, features, df_raw, metrics, test_comparison

# --- GIAO DIỆN CHÍNH ---
st.title("🚗 Hệ Thống Dự Đoán Giá Xe Ô Tô Cũ — Cây Quyết Định")
st.markdown("Thuật toán: **Decision Tree Regressor** | Tỷ lệ chia dữ liệu: **80% Train — 20% Test** | Thực hiện: **Nhóm 7**")

# Thanh bên điều chỉnh tham số cây quyết định
st.sidebar.header("🌳 Cấu hình Cây quyết định")
max_depth = st.sidebar.slider("Độ sâu tối đa của cây (max_depth)", min_value=3, max_value=25, value=10)

model, label_encoders, raw_options, features, df_raw, metrics, test_comparison = load_and_train_model(max_depth)

# --- PHẦN 1: CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST ---
st.subheader("📊 1. Đánh giá độ chính xác của mô hình trên tập Test")
st.caption(f"Số mẫu huấn luyện (Train): **{metrics['train_size']:,} xe** | Số mẫu kiểm thử (Test): **{metrics['test_size']:,} xe**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Hệ số xác định (R² Score)", f"{metrics['R2']:.4f}", f"{metrics['R2']*100:.2f}%")
m2.metric("Sai số tuyệt đối TB (MAE)", f"${metrics['MAE']:,.2f}")
m3.metric("Sai số bình phương TB (RMSE)", f"${metrics['RMSE']:,.2f}")
m4.metric("Sai số phần trăm TB (MAPE)", f"{metrics['MAPE']:.2f}%")

with st.expander("📋 Xem bảng đối chiếu kết quả trên 20 mẫu đầu của tập Test (80/20)"):
    st.dataframe(test_comparison, use_container_width=True)

st.markdown("---")

# --- PHẦN 2: DỰ ĐOÁN CHO 1 CHIẾC XE CỤ THỂ ---
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
    
    st.success("✅ Đã tính toán xong mức giá ước lượng từ Cây quyết định!")
    st.metric(label="Giá xe dự đoán (Predicted Price)", value=f"${predicted_price:,.2f} USD")

st.markdown("---")

# --- PHẦN 3: DỰ ĐOÁN HÀNG LOẠT TỪ FILE TEST NGOÀI ---
st.subheader("📂 3. Dự đoán hàng loạt từ file tập Test (CSV / Excel)")
uploaded_test = st.file_uploader("Tải lên file test.csv (hoặc .xlsx) của nhóm để dự đoán toàn bộ", type=['csv', 'xlsx'])

if uploaded_test is not None:
    df_test_ext = read_dataset(uploaded_test)
    if df_test_ext is not None:
        df_test_clean = clean_data(df_test_ext)
        
        # Kiểm tra xem có đủ 15 cột đầu vào không
        missing_cols = [c for c in features if c not in df_test_clean.columns]
        if missing_cols:
            st.error(f"⚠️ File tải lên bị thiếu các cột: {', '.join(missing_cols)}")
        else:
            X_ext = df_test_clean[features].copy()
            for col, le in label_encoders.items():
                X_ext[col] = X_ext[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            
            preds_ext = model.predict(X_ext)
            result_df = df_test_ext.copy()
            result_df['Predicted_Price ($)'] = np.round(preds_ext, 2)
            
            # Nếu file test ngoài có sẵn cột Price thực tế thì tính luôn độ chính xác
            if 'Price' in result_df.columns:
                y_true_ext = pd.to_numeric(result_df['Price'], errors='coerce').fillna(0)
                r2_ext = r2_score(y_true_ext, preds_ext)
                mae_ext = mean_absolute_error(y_true_ext, preds_ext)
                rmse_ext = np.sqrt(mean_squared_error(y_true_ext, preds_ext))
