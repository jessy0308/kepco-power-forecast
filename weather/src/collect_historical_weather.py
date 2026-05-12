import os
import requests
import json
import urllib.parse
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import time

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv('WEATHER_API_KEY')

# 기상청 종관기상관측(ASOS) 시간단위 데이터 조회 API
URL = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'

def fetch_asos_hourly(start_date, end_date, stn_id='108'):
    """
    지정된 기간 동안의 과거 기상 데이터(ASOS)를 수집합니다.
    API의 1회 호출 최대 건수가 999건이므로, 한 달 단위로 쪼개서 수집합니다.
    (1달 = 최대 744시간)
    """
    print(f"[{start_date} ~ {end_date}] 지점코드({stn_id}) 기상 데이터 수집 시작...\n")
    
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    all_items = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        # 이번 달의 마지막 날 계산
        next_month = current_dt + relativedelta(months=1)
        next_month_first_day = datetime(next_month.year, next_month.month, 1)
        month_end_dt = next_month_first_day - relativedelta(days=1)
        
        if month_end_dt > end_dt:
            month_end_dt = end_dt
            
        cur_start_str = current_dt.strftime('%Y%m%d')
        cur_end_str = month_end_dt.strftime('%Y%m%d')
        
        print(f"수집 중: {cur_start_str} ~ {cur_end_str} ... ", end="")
        
        params = {
            'serviceKey': API_KEY, # requests가 자동으로 인코딩 수행
            'pageNo': '1',
            'numOfRows': '999',
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'HR',
            'startDt': cur_start_str,
            'startHh': '00',
            'endDt': cur_end_str,
            'endHh': '23',
            'stnIds': str(stn_id)
        }
        
        max_retries = 3
        success = False
        for retry in range(max_retries):
            try:
                response = requests.get(URL, params=params, timeout=30)
                data = response.json()
                
                header = data.get('response', {}).get('header', {})
                if header.get('resultCode') == '00':
                    items = data['response']['body']['items']['item']
                    all_items.extend(items)
                    print(f"성공 ({len(items)}건)")
                    success = True
                    break
                else:
                    print(f"API 에러: {header.get('resultMsg')}")
                    break
            except Exception as e:
                print(f"예외 발생: {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"응답 내용: {response.text[:200]}")
                time.sleep(2)
        
        if not success:
            print("실패")
            
        time.sleep(0.5) # API 부하 방지
        current_dt = next_month_first_day

    return all_items

if __name__ == "__main__":
    # 전력수요 데이터 기간과 동일하게 2025년 전체 설정
    # stn_id '108'은 '서울'을 의미합니다. (전국을 대변하는 대표 지점으로 사용)
    items = fetch_asos_hourly('20250101', '20251231', '108')
    
    if items:
        df = pd.DataFrame(items)
        
        # 모델 학습에 필요한 주요 기상 컬럼 추출
        # tm: 시간, ta: 기온, rn: 강수량, ws: 풍속, hm: 습도
        cols_to_keep = ['tm', 'ta', 'rn', 'ws', 'hm']
        
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df_clean = df[existing_cols].copy()
        
        # 가독성을 위해 영문 이름으로 변경
        rename_dict = {
            'tm': 'datetime',
            'ta': 'temperature',
            'rn': 'rainfall',
            'ws': 'wind_speed',
            'hm': 'humidity'
        }
        df_clean.rename(columns=rename_dict, inplace=True)
        
        # 문자열 데이터를 숫자형으로 변환 (강수량 없는 경우 결측치 처리됨)
        for col in df_clean.columns:
            if col != 'datetime':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 강수량 결측치는 0으로 채움 (비가 안 온 시간)
        if 'rainfall' in df_clean.columns:
            df_clean['rainfall'] = df_clean['rainfall'].fillna(0)
            
        # 기온/습도 등의 단기 결측치는 앞뒤 시간 참조하여 선형 보간
        numeric_cols = [c for c in df_clean.columns if c != 'datetime']
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        
        # 저장 경로
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        file_path = os.path.join(data_dir, 'weather_historical_2025.csv')
        df_clean.to_csv(file_path, index=False)
        
        print(f"\n데이터 병합 및 전처리 완료! 총 {len(df_clean)}행 저장됨.")
        print(f"저장 경로: {os.path.abspath(file_path)}")
        print("\n--- 수집된 2025년 기상 데이터 상위 5행 ---")
        print(df_clean.head())
    else:
        print("데이터를 수집하지 못했습니다.")
