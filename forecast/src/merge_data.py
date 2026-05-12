import os
import pandas as pd

def merge_datasets():
    print("전력수요 데이터와 기상 데이터 병합을 시작합니다...\n")
    
    # 프로젝트 루트 디렉토리 탐색 (c:\wiset-elec-data)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    power_file = os.path.join(base_dir, 'kepco', 'data', 'power_demand_hourly.csv')
    weather_file = os.path.join(base_dir, 'weather', 'data', 'weather_historical_2025.csv')
    
    # 1. 전력수요 데이터 로드 및 datetime 변환
    if not os.path.exists(power_file):
        print(f"오류: 전력수요 파일이 존재하지 않습니다: {power_file}")
        return
    df_power = pd.read_csv(power_file)
    df_power['datetime'] = pd.to_datetime(df_power['datetime'])
    print(f"전력수요 데이터 로드 완료: {len(df_power)}행")
    
    # 2. 기상 데이터 로드 및 datetime 변환
    if not os.path.exists(weather_file):
        print(f"오류: 기상 데이터 파일이 존재하지 않습니다: {weather_file}")
        return
    df_weather = pd.read_csv(weather_file)
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    print(f"기상 데이터 로드 완료: {len(df_weather)}행")
    
    # 3. 데이터 병합 (Inner Join)
    # 일시(datetime)가 완벽히 일치하는 행만 결합하여 모델 학습용 데이터셋 구성
    df_merged = pd.merge(df_power, df_weather, on='datetime', how='inner')
    
    # 시간 순 정렬 및 인덱스 초기화
    df_merged.sort_values('datetime', inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    
    print(f"\n데이터 병합 성공! 총 {len(df_merged)}행 결합됨.")
    
    # 결측치 확인
    print("\n[결합된 데이터셋 결측치 현황]")
    print(df_merged.isnull().sum())
    
    # 새 프로젝트 폴더 규칙 준수: forecast 하위에 5개 폴더 생성 보장
    forecast_dir = os.path.join(base_dir, 'forecast')
    for folder in ['data', 'docs', 'src', 'images', 'report']:
        os.makedirs(os.path.join(forecast_dir, folder), exist_ok=True)
        
    # 결과 저장
    out_path = os.path.join(forecast_dir, 'data', 'merged_power_weather_2025.csv')
    df_merged.to_csv(out_path, index=False)
    
    print(f"\n최종 병합 데이터 저장 완료: {os.path.abspath(out_path)}")
    print("\n--- 병합된 데이터셋 상위 5행 미리보기 ---")
    print(df_merged.head())
    
    print("\n--- 주요 변수 요약 통계 ---")
    print(df_merged[['power_demand', 'temperature', 'humidity', 'wind_speed']].describe().round(2))

if __name__ == "__main__":
    merge_datasets()
