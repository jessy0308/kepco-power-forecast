import os
import pandas as pd
import matplotlib.pyplot as plt

try:
    from prophet import Prophet
    from prophet.serialize import model_to_json
except ImportError:
    Prophet = None

def train_and_evaluate():
    print("AI 기반 날씨 연계 전력수요 예측 모델(Prophet) 학습을 시작합니다...\n")
    
    # 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(base_dir, 'forecast', 'data', 'merged_power_weather_2025.csv')
    
    if not os.path.exists(data_file):
        print(f"오류: 병합된 데이터 파일이 없습니다: {data_file}")
        return
        
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"데이터 로드 완료: 총 {len(df)}행")
    
    # Prophet 입력 형식에 맞게 컬럼명 변경 (ds: 일시, y: 예측 대상)
    df_prophet = df[['datetime', 'power_demand', 'temperature', 'humidity', 'wind_speed']].copy()
    df_prophet.rename(columns={'datetime': 'ds', 'power_demand': 'y'}, inplace=True)
    
    if Prophet is None:
        print("\n[알림] prophet 라이브러리가 설치되어 있지 않습니다.")
        print("터미널에서 다음 명령어를 실행하여 설치해 주세요:")
        print("uv pip install prophet")
        return
        
    # 학습/테스트 데이터 분할 (마지막 7일을 테스트 셋으로 사용)
    test_days = 7
    test_hours = test_days * 24
    
    train_df = df_prophet.iloc[:-test_hours].copy()
    test_df = df_prophet.iloc[-test_hours:].copy()
    
    print(f"학습 데이터: {len(train_df)}행 / 테스트 데이터: {len(test_df)}행 (마지막 7일)")
    
    # 모델 정의 및 외생 변수(Regressor) 추가
    # 전력수요는 기온과 습도에 민감하므로 Regressor로 등록합니다.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95
    )
    model.add_regressor('temperature')
    model.add_regressor('humidity')
    model.add_regressor('wind_speed')
    
    print("\n모델 학습 진행 중...")
    model.fit(train_df, algorithm='LBFGS')
    print("모델 학습 완료!")
    
    # 학습된 모델 저장 (대시보드 실시간 예측용)
    models_dir = os.path.join(base_dir, 'forecast', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'prophet_model.json')
    with open(model_path, 'w') as fout:
        fout.write(model_to_json(model))
    print(f"모델 저장 완료: {os.path.abspath(model_path)}")

    
    # 예측을 위한 Future DataFrame 생성 (테스트 셋 기간 포함)
    # 외생 변수의 미래 값은 테스트 셋의 실제 기상 데이터를 매핑합니다.
    future = test_df[['ds', 'temperature', 'humidity', 'wind_speed']].copy()
    
    print("\n테스트 기간 예측 수행 중...")
    forecast = model.predict(future)
    
    # 성능 평가 (MAE, MAPE 계산)
    results = pd.merge(test_df[['ds', 'y']], forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    results['error'] = abs(results['y'] - results['yhat'])
    mae = results['error'].mean()
    mape = (results['error'] / results['y']).mean() * 100
    
    print("\n[예측 성능 평가 결과]")
    print(f"MAE (평균 절대 오차): {mae:.2f} MW")
    print(f"MAPE (평균 절대 백분율 오차): {mape:.2f}%")
    
    # 결과 시각화 및 이미지 저장 보장
    plt.figure(figsize=(12, 6))
    plt.plot(results['ds'], results['y'], label='Actual Demand', color='black', linewidth=1.5)
    plt.plot(results['ds'], results['yhat'], label='Predicted Demand (Prophet)', color='blue', linestyle='--')
    plt.fill_between(results['ds'], results['yhat_lower'], results['yhat_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('KEPCO Power Demand Forecast vs Actual (Last 7 Days)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Power Demand (MW)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    images_dir = os.path.join(base_dir, 'forecast', 'images')
    os.makedirs(images_dir, exist_ok=True)
    img_path = os.path.join(images_dir, 'demand_forecast_results.png')
    plt.savefig(img_path, dpi=300)
    print(f"\n예측 결과 그래프 저장 완료: {os.path.abspath(img_path)}")
    
    # 예측 결과 CSV 저장
    report_dir = os.path.join(base_dir, 'forecast', 'report')
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, 'forecast_evaluation.csv')
    results.to_csv(csv_path, index=False)
    print(f"예측 상세 데이터 저장 완료: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    train_and_evaluate()
