import os
import requests
import json
import urllib.parse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv('WEATHER_API_KEY')

# 초단기실황조회 API URL
URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'

def get_base_datetime():
    """현재 시간을 기준으로 기상청 API에서 요구하는 base_date, base_time 계산"""
    now = datetime.now()
    # 초단기실황은 매시간 40분에 생성되므로, 
    # 현재 시간이 40분 이전이면 한 시간 전 데이터를 호출
    if now.minute < 40:
        now = now - timedelta(hours=1)
    
    base_date = now.strftime('%Y%m%d')
    base_time = now.strftime('%H00')
    return base_date, base_time

def fetch_weather_data(nx=60, ny=127):
    """지정된 격자 좌표(nx, ny)의 기상 실황 데이터를 수집"""
    base_date, base_time = get_base_datetime()
    print(f"조회 기준 일시: {base_date} {base_time}, 좌표: ({nx}, {ny})")
    
    params = {
        'serviceKey': API_KEY,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': str(nx),
        'ny': str(ny)
    }
    
    response = requests.get(URL, params=params)
    
    try:
        data = response.json()
        return data
    except json.JSONDecodeError:
        print("JSON 파싱 에러. xml이 반환되었거나 응답 오류일 수 있습니다. 응답 데이터:")
        print(response.text)
        return None

if __name__ == "__main__":
    print("기상청 API 데이터 수집을 시작합니다...")
    data = fetch_weather_data()
    
    if data:
        header = data.get('response', {}).get('header', {})
        if header.get('resultCode') == '00':
            print("데이터 수집 성공!\n")
            
            # 저장할 폴더 설정 (weather/data/)
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # JSON 파일로 저장
            file_path = os.path.join(data_dir, 'weather_realtime.json')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            print(f"데이터가 성공적으로 저장되었습니다: {os.path.abspath(file_path)}\n")
            
            # 결과 일부 출력
            print("--- 현재 기상 실황 요약 ---")
            items = data['response']['body']['items']['item']
            
            # 항목별 이름 매핑
            category_names = {
                'T1H': '기온(℃)',
                'RN1': '1시간 강수량(mm)',
                'REH': '습도(%)',
                'PTY': '강수형태(0:없음,1:비,2:비/눈,3:눈,5:빗방울,6:빗방울눈날림,7:눈날림)',
                'VEC': '풍향(deg)',
                'WSD': '풍속(m/s)'
            }
            
            for item in items:
                cat = item['category']
                if cat in category_names:
                    print(f"- {category_names[cat]}: {item['obsrValue']}")
        else:
            print("API 호출 실패 (결과 코드 오류):")
            print(header.get('resultMsg', '알 수 없는 오류'))
    else:
        print("데이터를 가져오지 못했습니다.")
