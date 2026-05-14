import os
import json
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from prophet.serialize import model_from_json
except ImportError:
    print("Prophet 모듈이 설치되어 있지 않습니다.")
    exit(1)

def main():
    target_date = "20260513"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 5월 13일 어제 예측 데이터 vs 오늘 날씨 기반 예측 데이터 비교를 시작합니다.")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(base_dir, 'weather', '.env')
    load_dotenv(env_path)

    api_key = os.getenv('WEATHER_API_KEY')
    if not api_key:
        print("환경 변수(WEATHER_API_KEY)를 찾을 수 없습니다.")
        return

    # 1. 어제 예측 데이터 로드
    yesterday_pred_path = os.path.join(base_dir, 'forecast', 'data', 'daily_predictions', f'predict_{target_date}.csv')
    if not os.path.exists(yesterday_pred_path):
        print(f"어제 예측 데이터 파일을 찾을 수 없습니다: {yesterday_pred_path}")
        return
    
    df_yesterday = pd.read_csv(yesterday_pred_path)
    df_yesterday['ds'] = pd.to_datetime(df_yesterday['ds'])
    
    # 2. 오늘 날씨 데이터 조회를 위해 기상청 API 호출 (20260512 2300 기준 예보로 13일 전체 데이터 확보)
    decoded_key = urllib.parse.unquote(api_key) if '%' in api_key else api_key
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'
    
    base_date = "20260512"
    base_time = "2300"
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

    print(f"기상청 예보 데이터를 요청합니다... (조회 기준: {base_date} {base_time}, 대상일: {target_date})")
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data['response']['body']['items']['item']
    except Exception as e:
        print(f"기상청 API 데이터 수집 중 오류 발생: {e}")
        return

    forecast_dict = {}
    for item in items:
        if item['fcstDate'] == target_date:
            time_str = item['fcstTime'][:2] + ':00'
            dt_str = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]} {time_str}"
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

    df_today_weather = pd.DataFrame(list(forecast_dict.values()))
    if df_today_weather.empty:
        print("오늘 날씨 데이터를 파싱하지 못했습니다.")
        return
        
    df_today_weather['ds'] = pd.to_datetime(df_today_weather['ds'])
    df_today_weather = df_today_weather.sort_values('ds')

    # 3. 모델 로드 및 예측 수행
    model_path = os.path.join(base_dir, 'forecast', 'models', 'prophet_model.json')
    if not os.path.exists(model_path):
        print(f"예측 모델이 존재하지 않습니다: {model_path}")
        return

    print("학습된 Prophet 모델을 로드하여 오늘 날씨 기반 예측을 수행합니다...")
    with open(model_path, 'r') as fin:
        model = model_from_json(fin.read())
        
    forecast = model.predict(df_today_weather)
    df_today_weather['predicted_demand_MW'] = forecast['yhat'].values.round(2)

    # 4. 결과 비교 병합
    df_compare = pd.merge(
        df_yesterday, 
        df_today_weather, 
        on='ds', 
        suffixes=('_yest', '_today')
    )
    
    df_compare['demand_diff'] = (df_compare['predicted_demand_MW_today'] - df_compare['predicted_demand_MW_yest']).round(2)
    df_compare['temp_diff'] = (df_compare['temperature_today'] - df_compare['temperature_yest']).round(1)

    print("\n==================================== [ 5월 13일 예측 결과 비교 ] ====================================")
    print(f"{'시간':<12} | {'어제예측수요(MW)':<15} | {'오늘날씨반영수요(MW)':<16} | {'수요차이(MW)':<12} | {'어제온도':<8} | {'오늘온도':<8} | {'온도차이':<8}")
    print("-" * 105)
    for _, row in df_compare.iterrows():
        dt_str = row['ds'].strftime('%H:%M')
        print(f"{dt_str:<12} | {row['predicted_demand_MW_yest']:<18} | {row['predicted_demand_MW_today']:<19} | {row['demand_diff']:<14} | {row['temperature_yest']:<10} | {row['temperature_today']:<10} | {row['temp_diff']:<8}")
    print("-" * 105)
    
    # 요약 통계 출력
    mean_diff = df_compare['demand_diff'].mean()
    abs_mean_diff = df_compare['demand_diff'].abs().mean()
    print(f"\n[요약 분석]")
    print(f"- 평균 수요 차이 (오늘날씨반영 - 어제예측): {mean_diff:.2f} MW")
    print(f"- 평균 절대 오차 차이: {abs_mean_diff:.2f} MW")
    
    # 결과를 리포트 폴더에 CSV로 저장
    report_dir = os.path.join(base_dir, 'forecast', 'report')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'compare_result_20260513.csv')
    df_compare.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n비교 상세 결과가 저장되었습니다: {report_path}")

if __name__ == "__main__":
    main()
