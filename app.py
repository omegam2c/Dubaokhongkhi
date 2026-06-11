import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
import datetime
import math
import requests

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ---------------------------------------------------------
# CẤU HÌNH TRANG & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ thống Dự báo Ô nhiễm Không khí",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main background & typography */
        .reportview-container {
            background-color: #f4f6f9;
        }
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #1e293b;
        }
        
        /* Hourly Forecast Scroll */
        .hourly-forecast-container {
            display: flex;
            overflow-x: auto;
            gap: 15px;
            padding: 15px 5px;
            margin-bottom: 20px;
            scroll-behavior: smooth;
        }
        .hourly-forecast-container::-webkit-scrollbar {
            height: 8px;
        }
        .hourly-forecast-container::-webkit-scrollbar-track {
            background: #f1f1f1; 
            border-radius: 4px;
        }
        .hourly-forecast-container::-webkit-scrollbar-thumb {
            background: #cbd5e1; 
            border-radius: 4px;
        }
        .hourly-card {
            min-width: 110px;
            background: white;
            border-radius: 12px;
            padding: 15px 10px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #f1f5f9;
            flex-shrink: 0;
            transition: transform 0.2s;
        }
        .hourly-card:hover {
            transform: translateY(-3px);
        }
        .hourly-time { font-weight: 600; font-size: 1.1em; margin-bottom: 8px; color: #334155; }
        .hourly-pm { font-size: 1.4em; font-weight: 800; margin: 10px 0; padding: 6px; border-radius: 6px; color: white;}
        .hourly-weather { font-size: 0.9em; color: #64748b; margin-top: 5px; font-weight: 500;}
        
        /* PM2.5 Colors */
        .pm-level-1 { background-color: #10b981; color: white !important;} /* Good */
        .pm-level-2 { background-color: #facc15; color: #333 !important;} /* Moderate */
        .pm-level-3 { background-color: #fb923c; color: white !important;} /* Unhealthy for Sensitive */
        .pm-level-4 { background-color: #ef4444; color: white !important;} /* Unhealthy */
        .pm-level-5 { background-color: #a855f7; color: white !important;} /* Very Unhealthy */
        .pm-level-6 { background-color: #9f1239; color: white !important;} /* Hazardous */
        
        /* Health Recommendations Cards */
        .health-card {
            display: flex;
            align-items: center;
            background: white;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border: 1px solid #f8fafc;
        }
        .health-icon-wrapper {
            background-color: #fef3c7;
            color: #d97706;
            width: 54px;
            height: 54px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 18px;
            font-size: 26px;
            flex-shrink: 0;
        }
        .health-text {
            color: #334155;
            font-size: 1.05em;
            font-weight: 500;
        }
        .health-link {
            display: block;
            color: #3b82f6;
            font-size: 0.9em;
            text-decoration: none;
            margin-top: 4px;
            font-weight: 500;
        }
        .health-link:hover { text-decoration: underline; }
        
        /* Custom Button */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------
MODEL_PATH = "pj/best_global_CNN_BiLSTM_Attention.keras"
FEATURE_SCALER_PATH = "pj/feature_scaler.pkl"
TARGET_SCALER_PATH = "pj/target_scaler.pkl"
SEQ_LENGTH = 24
FEATURES = ['Temperature', 'Relative_Humidity', 'Wind_Speed', 'PM1', 'UM003', 'PM2.5', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
TARGET_INDEX = FEATURES.index('PM2.5')

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
if KERAS_AVAILABLE:
    @tf.keras.utils.register_keras_serializable(package="Custom")
    class TemporalAttention(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(TemporalAttention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                     initializer='glorot_uniform', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super(TemporalAttention, self).build(input_shape)

        def call(self, x):
            import tensorflow.keras.backend as K
            e = K.tanh(K.dot(x, self.W) + self.b)
            a = K.softmax(e, axis=1)
            output = x * a
            return K.sum(output, axis=1)
            
        def get_config(self):
            return super(TemporalAttention, self).get_config()
else:
    TemporalAttention = None

@st.cache_resource
def load_models():
    if not KERAS_AVAILABLE:
        st.error("❌ Thư viện TensorFlow/Keras chưa được cài đặt.")
        return None, None, None
        
    model, f_scaler, t_scaler = None, None, None
    try:
        model = load_model(MODEL_PATH, custom_objects={'TemporalAttention': TemporalAttention})
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình: {e}")
        
    try:
        with open(FEATURE_SCALER_PATH, 'rb') as f:
            f_scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Không thể tải Feature Scaler: {e}")
        
    try:
        with open(TARGET_SCALER_PATH, 'rb') as f:
            t_scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Không thể tải Target Scaler: {e}")
        
    return model, f_scaler, t_scaler

model, feature_scaler, target_scaler = load_models()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_pm_status(pm25_val):
    if pm25_val <= 12.0: return "pm-level-1", "Tốt"
    elif pm25_val <= 35.4: return "pm-level-2", "Trung bình"
    elif pm25_val <= 55.4: return "pm-level-3", "Kém"
    elif pm25_val <= 150.4: return "pm-level-4", "Xấu"
    elif pm25_val <= 250.4: return "pm-level-5", "Rất xấu"
    else: return "pm-level-6", "Nguy hại"

def generate_dummy_data():
    now = datetime.datetime.now()
    data = []
    for i in range(SEQ_LENGTH):
        t = now - datetime.timedelta(hours=SEQ_LENGTH - i)
        data.append({
            'Temperature': np.random.uniform(20, 35),
            'Relative_Humidity': np.random.uniform(50, 90),
            'Wind_Speed': np.random.uniform(0, 5),
            'PM1': np.random.uniform(10, 50),
            'UM003': np.random.uniform(10, 50),
            'PM2.5': np.random.uniform(15, 80),
            'hour_sin': math.sin(2 * math.pi * t.hour / 24),
            'hour_cos': math.cos(2 * math.pi * t.hour / 24),
            'month_sin': math.sin(2 * math.pi * t.month / 12),
            'month_cos': math.cos(2 * math.pi * t.month / 12)
        })
    return pd.DataFrame(data)[FEATURES]

def pad_data(df, target_len=SEQ_LENGTH):
    if len(df) >= target_len:
        return df.iloc[-target_len:].copy()
    else:
        diff = target_len - len(df)
        pad_df = pd.DataFrame([df.iloc[0]] * diff)
        return pd.concat([pad_df, df], ignore_index=True)

def preprocess_csv_data(df):
    """Tự động nội suy các cột thời gian (sin/cos) và PM1, UM003 nếu file CSV tải lên bị thiếu"""
    
    # Chuẩn hóa tên cột PM2.5 (Phòng trường hợp viết hoa/thường hoặc thiếu dấu chấm)
    for col in df.columns:
        if str(col).strip().lower() in ['pm2.5', 'pm25', 'pm 2.5']:
            df.rename(columns={col: 'PM2.5'}, inplace=True)
            break
            
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    if 'hour_sin' in missing_features:
        # Tìm cột thời gian
        time_col = None
        for col in ['time', 'date', 'datetime', 'Ngày', 'Time', 'Date']:
            if col in df.columns:
                time_col = col
                break
                
        if time_col:
            dt_series = pd.to_datetime(df[time_col])
        else:
            # Tạo chuỗi thời gian giả lập tính từ hiện tại lùi về trước
            now = datetime.datetime.now()
            dt_series = pd.Series([now + datetime.timedelta(hours=i) for i in range(len(df))])
            
        if 'hour_sin' not in df.columns: df['hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
        if 'hour_cos' not in df.columns: df['hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
        if 'month_sin' not in df.columns: df['month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
        if 'month_cos' not in df.columns: df['month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
            
    # Tự nội suy PM1 và UM003 từ PM2.5 theo tỷ lệ giả định nếu thiếu
    if 'PM1' not in df.columns and 'PM2.5' in df.columns:
        df['PM1'] = df['PM2.5'] * 0.7
    if 'UM003' not in df.columns and 'PM2.5' in df.columns:
        df['UM003'] = df['PM2.5'] * 0.6
        
    # Điền 0 cho các cột còn thiếu khác để tránh lỗi IndexError
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
            
    return df

COORDS = {
    "Đống Đa": {"lat": 21.0123, "lon": 105.8211},
    "Cầu Giấy": {"lat": 21.0362, "lon": 105.7906},
    "Sóc Sơn": {"lat": 21.2581, "lon": 105.8164}
}

def get_open_meteo_data(lat, lon):
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&past_days=1"
    air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&past_days=1"
    
    weather_resp = requests.get(weather_url)
    air_resp = requests.get(air_url)
    
    if weather_resp.status_code != 200:
        raise Exception(f"Lỗi API Thời tiết ({weather_resp.status_code}): Quá tải hoặc bị chặn")
    if air_resp.status_code != 200:
        raise Exception(f"Lỗi API Không khí ({air_resp.status_code}): Quá tải hoặc bị chặn")
        
    weather_resp = weather_resp.json()
    air_resp = air_resp.json()
    
    w_hourly = weather_resp.get('hourly', {})
    a_hourly = air_resp.get('hourly', {})
    
    df = pd.DataFrame({
        'Temperature': w_hourly.get('temperature_2m', [0]*24)[-24:],
        'Relative_Humidity': w_hourly.get('relative_humidity_2m', [0]*24)[-24:],
        'Wind_Speed': w_hourly.get('wind_speed_10m', [0]*24)[-24:],
        'PM2.5': a_hourly.get('pm2_5', [0]*24)[-24:],
        'time': w_hourly.get('time', [datetime.datetime.now().isoformat()]*24)[-24:]
    })
    
    df['PM1'] = df['PM2.5'] * 0.7
    df['UM003'] = df['PM2.5'] * 0.6
    
    df['time'] = pd.to_datetime(df['time'])
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
    
    df = df.bfill().ffill().fillna(0)
    return df[FEATURES]

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------
def render_health_recommendations(pm_val):
    st.markdown("### Khuyến nghị về sức khoẻ")
    
    recs = []
    if pm_val > 35.4: # Kém trở lên
        recs.append(("🚴", "Các nhóm nhạy cảm nên giảm tập thể dục ngoài trời", ""))
        recs.append(("🪟", "Đóng cửa sổ để tránh không khí bẩn bên ngoài", "Mua một trình theo dõi"))
        recs.append(("😷", "Các nhóm nhạy cảm nên đeo mặt nạ khi ra ngoài", "Mua Mặt nạ"))
        recs.append(("🌬️", "Các nhóm nhạy cảm nên khởi động máy lọc không khí", "Mua máy lọc không khí"))
    elif pm_val > 12.0: # Trung bình
        recs.append(("🚴", "Các nhóm nhạy cảm nên hạn chế hoạt động quá sức", ""))
        recs.append(("🪟", "Có thể mở cửa sổ, nhưng theo dõi chất lượng không khí", ""))
    else: # Tốt
        recs.append(("🏃", "Thời tiết lý tưởng để hoạt động ngoài trời", ""))
        recs.append(("🪟", "Mở cửa sổ để đón không khí trong lành", ""))
        
    html = ""
    for icon, text, link in recs:
        link_html = f"<a href='#' class='health-link'>{link}</a>" if link else ""
        html += f"<div class='health-card'><div class='health-icon-wrapper'>{icon}</div><div><div class='health-text'>{text}</div>{link_html}</div></div>"
    st.markdown(html, unsafe_allow_html=True)

def render_pm_legend():
    st.markdown("""
    <style>
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 10px;
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f8fafc;
    }
    .legend-item {
        display: flex;
        align-items: center;
        font-size: 0.95em;
        font-weight: 600;
        color: #334155;
    }
    .legend-color {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        margin-right: 8px;
    }
    </style>
    <div class="legend-container">
        <div class="legend-item"><div class="legend-color pm-level-1"></div>Tốt (0 - 12)</div>
        <div class="legend-item"><div class="legend-color pm-level-2"></div>Trung bình (12.1 - 35.4)</div>
        <div class="legend-item"><div class="legend-color pm-level-3"></div>Kém (35.5 - 55.4)</div>
        <div class="legend-item"><div class="legend-color pm-level-4"></div>Xấu (55.5 - 150.4)</div>
        <div class="legend-item"><div class="legend-color pm-level-5"></div>Rất xấu (150.5 - 250.4)</div>
        <div class="legend-item"><div class="legend-color pm-level-6"></div>Nguy hại (> 250.4)</div>
    </div>
    """, unsafe_allow_html=True)

# Đã gỡ bỏ hàm get_empirical_metrics và render_evaluation_table theo yêu cầu

def render_hourly_forecast(predictions, df_input, h_steps):
    st.markdown("### Dự báo theo giờ")
    st.markdown("Dự báo mức độ PM2.5 trong các giờ tới")
    
    render_pm_legend()
    
    now = datetime.datetime.now()
    html = '<div class="hourly-forecast-container">'
    
    for i, pred in enumerate(predictions):
        future_time = now + datetime.timedelta(hours=i+1)
        time_str = future_time.strftime("%H:00")
        color_class, label = get_pm_status(pred)
        
        html += f"<div class='hourly-card'>" \
                f"<div class='hourly-time'>{time_str}</div>" \
                f"<div class='hourly-pm {color_class}'>{int(pred)}</div>" \
                f"<div class='hourly-weather' style='font-weight: 600; color: #334155; font-size: 1em;'>{label}</div>" \
                f"</div>"
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR / CÀI ĐẶT
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208945.png", width=100)
    st.title("⚙️ Cấu hình Dự báo")
    
    forecast_mode = st.radio(
        "⏱️ Chế độ dự báo:",
        ["Nhiều bước (Multi-step)", "1 bước (Single-step)"]
    )
    
    h_steps = 1
    if forecast_mode == "Nhiều bước (Multi-step)":
        h_steps = st.slider("Số bước dự báo (Giờ):", min_value=2, max_value=24, value=15)
        
    st.markdown("---")
    st.markdown("### Về Ứng dụng")
    st.markdown("Hệ thống dự báo PM2.5 dựa trên mô hình Deep Learning (LSTM-Attention).")

# ---------------------------------------------------------
# GIAO DIỆN CHÍNH (TABS)
# ---------------------------------------------------------
st.title("🌍 Dashboard Dự báo Ô nhiễm Không khí")
st.markdown("Đo lường và dự báo bụi mịn PM2.5 cho các khu vực tại Hà Nội.")

tabs = st.tabs(["🌐 Dự báo Thực tế", "📈 Đánh giá Mô hình (Backtest)"])

# ---------------------------------------------------------
# TAB 1: DỰ BÁO THỰC TẾ
# ---------------------------------------------------------
with tabs[0]:
    locations = ["Đống Đa", "Cầu Giấy", "Sóc Sơn"]
    
    col_loc, col_empty = st.columns([1, 2])
    with col_loc:
        loc_name = st.selectbox("📍 Chọn khu vực cần dự báo:", locations)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"#### 📥 Dữ liệu thời gian thực - {loc_name}")
        st.info(f"Hệ thống sẽ tự động đồng bộ dữ liệu thời tiết và không khí 24 giờ qua từ API Open-Meteo cho khu vực {loc_name}.")
        
    input_df = pd.DataFrame(columns=FEATURES)

    if st.button(f"🔄 Đồng bộ dữ liệu 24h qua tại {loc_name}", key="sync_api"):
        with st.spinner("Đang tải dữ liệu..."):
            try:
                # Dùng tọa độ mặc định nếu khu vực không có trong list COORDS
                lat = COORDS.get(loc_name, {"lat": 21.0285})["lat"]
                lon = COORDS.get(loc_name, {"lon": 105.8542})["lon"]
                df_api = get_open_meteo_data(lat, lon)
                st.session_state['realtime_df'] = df_api
                st.success("Tải dữ liệu thành công!")
            except Exception as e:
                st.error(f"Lỗi khi lấy dữ liệu: {e}")
    
    if 'realtime_df' in st.session_state:
        input_df = st.session_state['realtime_df']
        with st.expander("Xem dữ liệu đầu vào", expanded=False):
            st.dataframe(input_df, use_container_width=True)

    st.markdown("---")
    run_forecast = st.button(f"🚀 Thực hiện Dự báo cho {loc_name}", type="primary", use_container_width=True, key="run_forecast")

    if run_forecast:
        if model is None:
            st.error("❌ Mô hình chưa được nạp.")
        elif input_df.empty or len(input_df) < SEQ_LENGTH:
            st.warning(f"⚠️ Vui lòng cung cấp đủ {SEQ_LENGTH} dòng dữ liệu.")
        else:
            with st.spinner(f"Đang chạy mô hình cho {loc_name}..."):
                try:
                    scaled_input = feature_scaler.transform(input_df.values)
                    current_seq = scaled_input.copy()
                    predictions = []
                    
                    for step in range(h_steps):
                        X_infer = current_seq.reshape(1, SEQ_LENGTH, len(FEATURES))
                        pred_scaled = model.predict(X_infer, verbose=0)
                        pred_val = target_scaler.inverse_transform(pred_scaled)[0][0]
                        predictions.append(max(0, pred_val))
                        
                        if h_steps > 1:
                            new_row = current_seq[-1].copy()
                            new_row[TARGET_INDEX] = pred_scaled[0][0]
                            current_seq = np.vstack([current_seq[1:], new_row])

                    st.success(f"✅ Dự báo hoàn tất cho {loc_name}!")
                    
                    st.markdown("---")
                    render_hourly_forecast(predictions, input_df, h_steps)
                    st.markdown("---")
                    max_pm = max(predictions)
                    render_health_recommendations(max_pm)
                    
                except Exception as e:
                    st.error(f"Lỗi dự báo: {e}")

# ---------------------------------------------------------
# TAB 2: BACKTEST TAB
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 📈 Đánh giá Mô hình bằng Dữ liệu Thực tế (Backtest)")
    st.markdown("Tải lên file dữ liệu Test (chứa nhiều khu vực) hoặc nhập tay để chạy dự báo và đánh giá độ tin cậy.")
    
    input_method_bt = st.selectbox("Phương thức nhập liệu:", ["Tải lên file CSV", "Nhập tay"], key="bt_method")
    
    df_test = None
    
    if input_method_bt == "Tải lên file CSV":
        test_file = st.file_uploader("Tải lên file CSV dữ liệu Test (VD: exact_test_data.csv)", type=["csv"], key="backtest_up")
        if test_file is not None:
            try:
                df_test = pd.read_csv(test_file)
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")
    else:
        st.info("Nhập số liệu vào bảng dưới đây (Tối thiểu 25 dòng để có thể tạo chuỗi 24h và đối chiếu 1h tiếp theo):")
        cols = ['Timestamp', 'District_ID'] + FEATURES
        empty_df = pd.DataFrame([["2023-01-01 00:00:00", "Khu_vuc_1"] + [0]*len(FEATURES)]*25, columns=cols)
        edited_df = st.data_editor(empty_df, num_rows="dynamic", use_container_width=True, key="ed_manual_bt")
        if not edited_df.empty and len(edited_df) >= 25:
            df_test = edited_df.copy()
            
    if df_test is not None:
        try:
            df_test = preprocess_csv_data(df_test)
            st.success(f"Đã tải thành công file chứa {len(df_test)} dòng dữ liệu.")
            
            if st.button("🚀 Thực hiện Đánh giá (Backtest)", type="primary", use_container_width=True):
                all_results = []
                
                if 'District_ID' not in df_test.columns:
                    df_test['District_ID'] = 'Chung'
                    
                districts = df_test['District_ID'].unique()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_districts = len(districts)
                
                for i, district in enumerate(districts):
                    status_text.text(f"Đang xử lý dự báo cho khu vực: {district} ({i+1}/{total_districts})...")
                    df_dist = df_test[df_test['District_ID'] == district]
                    
                    # Sắp xếp theo Timestamp nếu có
                    if 'Timestamp' in df_dist.columns:
                        df_dist = df_dist.sort_values('Timestamp')
                    
                    if len(df_dist) <= SEQ_LENGTH:
                        continue
                        
                    features_data = df_dist[FEATURES].values
                    scaled_data = feature_scaler.transform(features_data)
                    
                    num_samples = len(scaled_data) - SEQ_LENGTH
                    X_batch = np.array([scaled_data[j:j+SEQ_LENGTH] for j in range(num_samples)])
                    y_true_actual = features_data[SEQ_LENGTH:, TARGET_INDEX]
                    
                    if 'Timestamp' in df_dist.columns:
                        timestamps = df_dist['Timestamp'].values[SEQ_LENGTH:]
                    else:
                        timestamps = np.arange(SEQ_LENGTH, len(df_dist))
                    
                    # Vectorized prediction
                    preds_scaled = model.predict(X_batch, batch_size=64, verbose=0)
                    preds_actual = target_scaler.inverse_transform(preds_scaled).flatten()
                    preds_actual = np.maximum(0, preds_actual) # Loại bỏ giá trị âm
                    
                    df_res = pd.DataFrame({
                        'District_ID': district,
                        'Thời gian': timestamps,
                        'Thực tế': y_true_actual,
                        'Dự báo': preds_actual
                    })
                    all_results.append(df_res)
                    progress_bar.progress((i + 1) / total_districts)
                    
                status_text.text("✅ Đã hoàn tất dự báo cho tất cả khu vực!")
                progress_bar.empty()
                
                if len(all_results) > 0:
                    st.session_state['backtest_results'] = pd.concat(all_results, ignore_index=True)
                else:
                    st.error("Dữ liệu quá ngắn, không đủ để tạo chuỗi.")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
            
    # Hiển thị Dashboard Đánh giá nếu đã có kết quả trong session
    if 'backtest_results' in st.session_state:
        results_df = st.session_state['backtest_results']
        
        # 1. GLOBAL METRICS
        st.markdown("---")
        st.markdown("### 🌍 Bức Tranh Toàn Cảnh (Global Metrics)")
        st.markdown("Độ chính xác của mô hình tính trên **tổng thể toàn bộ các khu vực**.")
        
        y_true_global = results_df['Thực tế'].values
        y_pred_global = results_df['Dự báo'].values
        
        r2_global = r2_score(y_true_global, y_pred_global) if np.var(y_true_global) > 0 else 0
        mae_global = mean_absolute_error(y_true_global, y_pred_global)
        rmse_global = math.sqrt(mean_squared_error(y_true_global, y_pred_global))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng R² Score", f"{r2_global:.4f}", help="Gần 1.0 là xuất sắc")
        c2.metric("Tổng MAE", f"{mae_global:.2f} µg/m³", help="Sai số tuyệt đối trung bình")
        c3.metric("Tổng RMSE", f"{rmse_global:.2f} µg/m³", help="Sai số bình phương trung bình")
        
        # 1.5 SO SÁNH ĐA KHU VỰC (COMPARATIVE DASHBOARD)
        st.markdown("---")
        st.markdown("### 🏆 Bảng Xếp hạng & So sánh Xu hướng Ô nhiễm")
        
        # Chuẩn bị dữ liệu cho Bảng xếp hạng và Biểu đồ
        districts_list = results_df['District_ID'].unique().tolist()
        ranking_data = []
        plot_data_list = []
        
        for d in districts_list:
            df_d = results_df[results_df['District_ID'] == d]
            preds = df_d['Dự báo'].values
            
            # Leaderboard metrics
            avg_pm = np.mean(preds)
            max_pm = np.max(preds)
            color_class, status_text = get_pm_status(avg_pm)
            ranking_data.append({
                'Khu vực': d,
                'Trung bình PM2.5 (µg/m³)': avg_pm,
                'Cao nhất PM2.5 (µg/m³)': max_pm,
                'Mức độ TB': status_text
            })
            
            # Lấy 100 điểm cuối cho biểu đồ
            df_plot = df_d.tail(100).copy()
            df_plot['Index'] = range(len(df_plot)) # Đồng bộ trục X
            plot_data_list.append(df_plot[['Index', 'District_ID', 'Dự báo']])
                
        df_ranking = pd.DataFrame(ranking_data).sort_values(by='Trung bình PM2.5 (µg/m³)', ascending=False).reset_index(drop=True)
        df_ranking.index = df_ranking.index + 1
        
        df_multi_line = pd.concat(plot_data_list, ignore_index=True) if len(plot_data_list) > 0 else pd.DataFrame()
        
        chart_multi = alt.Chart(df_multi_line).mark_line().encode(
            x=alt.X('Index:O', title='Thời gian (100 mẫu gần nhất)'),
            y=alt.Y('Dự báo:Q', title='PM2.5 (µg/m³)', scale=alt.Scale(zero=False)),
            color=alt.Color('District_ID:N', legend=alt.Legend(title="Khu vực", orient="bottom")),
            tooltip=['District_ID', 'Index', alt.Tooltip('Dự báo:Q', format='.2f')]
        ).properties(height=350)
        
        col_rank, col_chart = st.columns([1, 1.5])
        with col_rank:
            st.markdown("#### Ô nhiễm Nhất (Leaderboard)")
            st.dataframe(df_ranking.style.format({"Trung bình PM2.5 (µg/m³)": "{:.2f}", "Cao nhất PM2.5 (µg/m³)": "{:.2f}"}), use_container_width=True)
            
        with col_chart:
            st.markdown("#### Biểu đồ Xu hướng")
            st.altair_chart(chart_multi, use_container_width=True)

        # 2. LOCAL METRICS & CHARTS
        st.markdown("---")
        st.markdown("### 📍 Phân tích Chuyên sâu từng Khu vực (Local Analysis)")
        
        districts_av = results_df['District_ID'].unique().tolist()
        selected_dist = st.selectbox("🔍 Chọn khu vực để xem chi tiết:", districts_av)
        
        dist_df = results_df[results_df['District_ID'] == selected_dist]
        
        y_true_local = dist_df['Thực tế'].values
        y_pred_local = dist_df['Dự báo'].values
        
        r2_local = r2_score(y_true_local, y_pred_local) if np.var(y_true_local) > 0 else 0
        mae_local = mean_absolute_error(y_true_local, y_pred_local)
        rmse_local = math.sqrt(mean_squared_error(y_true_local, y_pred_local))
        
        st.markdown(f"**Kết quả đánh giá tại `{selected_dist}` (Số mẫu: {len(dist_df)}):**")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("R² Score", f"{r2_local:.4f}")
        lc2.metric("MAE", f"{mae_local:.2f} µg/m³")
        lc3.metric("RMSE", f"{rmse_local:.2f} µg/m³")
        
        st.markdown("---")
        # 3. HOURLY FORECAST & HEALTH RECOMMENDATIONS
        # Lấy 15 giá trị dự báo cuối cùng để hiển thị (tương đương 15 giờ)
        last_predictions = dist_df['Dự báo'].tail(15).tolist()
        
        if len(last_predictions) > 0:
            render_hourly_forecast(last_predictions, None, len(last_predictions))
            
            st.markdown("---")
            max_pm = max(last_predictions)
            render_health_recommendations(max_pm)

