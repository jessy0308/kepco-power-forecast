import os
import json
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import requests

try:
    from prophet.serialize import model_from_json
except ImportError:
    print("Prophet 모듈이 설치되어 있지 않습니다. 'uv pip install prophet' 을 실행해주세요.")
    exit(1)

from dotenv import load_dotenv

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 일일 전력수요 예측 자동화 스크립트를 시작합니다.")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(base_dir, 'weather', '.env')
    load_dotenv(env_path)

    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        print("환경 변수(WEATHER_API_KEY)를 찾을 수 없습니다.")
        return

    # 단기예보 기준 시간 계산 로직
    now = datetime.now()
    today = now.strftime('%Y%m%d')
    tomorrow_date = (now + timedelta(days=1)).strftime('%Y%m%d')
    
    times = [2, 5, 8, 11, 14, 17, 20, 23]
    base_date = today
    base_time = "0200"
    for t in reversed(times):
        if now.hour > t or (now.hour == t and now.minute >= 15):
            base_time = f"{t:02d}00"
            break
    else:
        yesterday = now - timedelta(days=1)
        base_date = yesterday.strftime('%Y%m%d')
        base_time = "2300"

    decoded_key = urllib.parse.unquote(api_key) if '%' in api_key else api_key
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    params = {
        'serviceKey': decoded_key,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': '60',
        'ny': '127'
    }

    print(f"기상청 예보 데이터를 요청합니다... (조회 기준: {base_date} {base_time})")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data['response']['body']['items']['item']
    except Exception as e:
        print(f"단기예보 수집 중 오류가 발생했습니다: {e}")
        return

    # 내일 날씨만 추출
    forecast_dict = {}
    for item in items:
        if item['fcstDate'] == tomorrow_date:
            time_str = item['fcstTime'][:2] + ':00'
            dt_str = f"{tomorrow_date[:4]}-{tomorrow_date[4:6]}-{tomorrow_date[6:]} {time_str}"
            if dt_str not in forecast_dict:
                forecast_dict[dt_str] = {'ds': dt_str}
            
            cat = item['category']
            if cat in ['TMP', 'REH', 'WSD']:
                try:
                    val = float(item['fcstValue'])
                    if cat == 'TMP': forecast_dict[dt_str]['temperature'] = val
                    elif cat == 'REH': forecast_dict[dt_str]['humidity'] = val
                    elif cat == 'WSD': forecast_dict[dt_str]['wind_speed'] = val
                except ValueError:
                    pass

    df_future = pd.DataFrame(list(forecast_dict.values()))
    if df_future.empty:
        print("내일의 날씨 데이터를 파싱하지 못했습니다. (데이터 누락)")
        return
        
    df_future['ds'] = pd.to_datetime(df_future['ds'])
    df_future = df_future.sort_values('ds')

    # 모델 불러오기 및 예측
    model_path = os.path.join(base_dir, 'forecast', 'models', 'prophet_model.json')
    if not os.path.exists(model_path):
        print(f"예측 모델이 존재하지 않습니다: {model_path}")
        return

    print("학습된 Prophet 모델을 로드하여 예측을 수행합니다...")
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(fin.read())
            
        forecast = model.predict(df_future)
        
        # 결과 정리
        df_result = df_future.copy()
        df_result['predicted_demand_MW'] = forecast['yhat'].values.round(2)
        
        # 저장할 디렉토리 생성
        save_dir = os.path.join(base_dir, 'forecast', 'data', 'daily_predictions')
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 일일 백업용 CSV 저장 (파일명: predict_20260513.csv 형식)
        save_path = os.path.join(save_dir, f'predict_{tomorrow_date}.csv')
        df_result.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        # 2. 마스터 데이터 파일에 누적 (Append)
        master_path = os.path.join(base_dir, 'forecast', 'data', 'all_predictions_master.csv')
        if os.path.exists(master_path):
            df_result.to_csv(master_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_result.to_csv(master_path, index=False, encoding='utf-8-sig')
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 완료! 일일 백업({os.path.basename(save_path)}) 및 마스터 누적이 완료되었습니다.")

    except Exception as e:
         print(f"예측 모델 구동 또는 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
