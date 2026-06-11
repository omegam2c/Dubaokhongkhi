import requests
import pandas as pd
import requests

def download_full_historical_data():
    DISTRICT_ID = "Thanh_Xuan"
    LAT = 20.9947
    LON = 105.7998
    AREA_TYPE = 2
    POP_DENSITY = 23000
    
    # Khoảng thời gian muốn lấy 
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    print(f"Fetching historical data for {DISTRICT_ID} from {start_date} to {end_date}...")
    
    # 2. Lấy dữ liệu từ Open-Meteo Archive API
    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=auto"
    air_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}&start_date={start_date}&end_date={end_date}&hourly=pm2_5&timezone=auto"
    
    try:
        w_data = requests.get(weather_url).json()['hourly']
        a_data = requests.get(air_url).json()['hourly']
        
        # 3. Tạo DataFrame 
        df = pd.DataFrame()
        
        # Cột thời gian
        df['Timestamp'] = pd.to_datetime(w_data['time'])
        
        # Cột tĩnh
        df['District_ID'] = DISTRICT_ID
        df['Latitude'] = LAT
        df['Longitude'] = LON
        df['Area_Type'] = AREA_TYPE
        df['Pop_Density'] = POP_DENSITY
        
        # Cột thời tiết và PM2.5
        df['Temperature'] = w_data['temperature_2m']
        df['Relative_Humidity'] = w_data['relative_humidity_2m']
        df['Wind_Speed'] = w_data['wind_speed_10m']
        df['PM2.5'] = a_data['pm2_5']
        
        # Cột PM1 và UM003 
        df['PM1'] = df['PM2.5'] * 0.7
        df['UM003'] = df['PM2.5'] * 39.7
        
        # Cột Feature Engineering (Tách từ Timestamp)
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Month'] = df['Timestamp'].dt.month
        
        # Sắp xếp lại thứ tự cột cho giống hệt file exact_test_data.csv
        COLUMNS_ORDER = [
            'Timestamp', 'District_ID', 'Latitude', 'Longitude', 'Area_Type', 
            'Pop_Density', 'Temperature', 'Relative_Humidity', 'Wind_Speed', 
            'PM1', 'PM2.5', 'UM003', 'Hour', 'DayOfWeek', 'Month'
        ]
        df = df[COLUMNS_ORDER]
        
        # 4. Lưu ra file CSV
        output_file = "test_data_full.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDone! File {output_file} has been created successfully.")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_full_historical_data()