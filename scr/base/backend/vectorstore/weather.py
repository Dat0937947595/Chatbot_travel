import requests

def get_weather(city_name):
    api_key = "b13f85eb589c453522bb1322a6763a8d" 
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    # Xây dựng URL đầy đủ
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    # Gửi yêu cầu HTTP GET tới API
    response = requests.get(complete_url)
    
    # Chuyển dữ liệu phản hồi thành định dạng JSON
    data = response.json()
    
    # Kiểm tra trạng thái phản hồi
    if data["cod"] != "404":
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        temperature = main["temp"]
        
        print(f"Thời tiết ở {city_name}: {weather_desc}")
        print(f"Nhiệt độ: {temperature}°C")
    else:
        print("Không tìm thấy thành phố.")

# Gọi hàm
#get_weather("Ho Chi Minh City")