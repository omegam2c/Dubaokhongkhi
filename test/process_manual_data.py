import pandas as pd
import numpy as np

def fill_missing_attributes(input_csv, output_csv):
    """
    Đọc file CSV chứa dữ liệu thủ công (chỉ có Nhiệt độ, Độ ẩm, Gió, PM2.5)
    và tự động tính toán/điền thêm các cột còn thiếu để đủ 15 cột chuẩn.
    """
    print(f"Reading manual data file: {input_csv} ...")
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found!")
        return

    # 1. Từ điển thông tin tĩnh (Tọa độ, Dân số, Khu vực) của các quận
    # Nếu file của bạn là quận khác, bạn có thể thêm thông tin vào đây
    DISTRICT_INFO = {
        "Cau_Giay": {"lat": 21.036, "lon": 105.795, "area": 2, "pop": 23000},
        "Dong_Da": {"lat": 21.012, "lon": 105.821, "area": 2, "pop": 37000},
        "Soc_Son": {"lat": 21.258, "lon": 105.816, "area": 1, "pop": 1100},
        "Thanh_Xuan": {"lat": 20.994, "lon": 105.799, "area": 2, "pop": 31000}
    }
    
    # 2. Xử lý thời gian (Tạo Hour, DayOfWeek, Month)
    # Cột thời gian có thể là 'Timestamp' hoặc 'time', ta chuẩn hóa lại thành 'Timestamp'
    time_col = 'time' if 'time' in df.columns else 'Timestamp'
    df['Timestamp'] = pd.to_datetime(df[time_col])
    
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    # 3. Tính toán PM1 và UM003 (Nội suy từ PM2.5)
    if 'PM2.5' not in df.columns and 'pm2_5' in df.columns:
        df['PM2.5'] = df['pm2_5'] # Sửa lại tên cột nếu tên lấy từ API bị thường
        
    df['PM1'] = df['PM2.5'] * 0.7
    df['UM003'] = df['PM2.5'] * 39.7

    # 4. Điền thông tin tĩnh dựa vào District_ID
    # Giả định mặc định là Cầu Giấy nếu file không có cột District_ID
    if 'District_ID' not in df.columns:
        df['District_ID'] = "Cau_Giay"
        
    def get_info(row, key):
        dist = row['District_ID']
        return DISTRICT_INFO.get(dist, DISTRICT_INFO["Cau_Giay"])[key]

    df['Latitude'] = df.apply(lambda row: get_info(row, 'lat'), axis=1)
    df['Longitude'] = df.apply(lambda row: get_info(row, 'lon'), axis=1)
    df['Area_Type'] = df.apply(lambda row: get_info(row, 'area'), axis=1)
    df['Pop_Density'] = df.apply(lambda row: get_info(row, 'pop'), axis=1)

    # Đảm bảo tên cột Thời tiết cũng đúng chuẩn
    if 'Temperature' not in df.columns and 'temperature_2m' in df.columns:
        df['Temperature'] = df['temperature_2m']
    if 'Relative_Humidity' not in df.columns and 'relative_humidity_2m' in df.columns:
        df['Relative_Humidity'] = df['relative_humidity_2m']
    if 'Wind_Speed' not in df.columns and 'wind_speed_10m' in df.columns:
        df['Wind_Speed'] = df['wind_speed_10m']

    # 5. Lọc và sắp xếp chuẩn đúng 15 cột bài yêu cầu
    COLUMNS_ORDER = [
        'Timestamp', 'District_ID', 'Latitude', 'Longitude', 'Area_Type', 
        'Pop_Density', 'Temperature', 'Relative_Humidity', 'Wind_Speed', 
        'PM1', 'PM2.5', 'UM003', 'Hour', 'DayOfWeek', 'Month'
    ]
    
    # Chỉ giữ lại đúng 15 cột này
    final_df = df[COLUMNS_ORDER]
    
    # Lưu ra file mới
    final_df.to_csv(output_csv, index=False)
    print(f"\nSUCCESS! Created {output_csv} with 15 columns.")
    print("First 5 rows:")
    print(final_df.head())

if __name__ == "__main__":
    # BẠN HÃY SỬA TÊN FILE TẠI ĐÂY NẾU CẦN
    INPUT_FILE = "test_data_caugiay.csv"   # File bạn tự tải bằng tay thiếu cột
    OUTPUT_FILE = "test_data_caugiay_full_attributes.csv" # File sẽ được tạo ra
    
    fill_missing_attributes(INPUT_FILE, OUTPUT_FILE)
