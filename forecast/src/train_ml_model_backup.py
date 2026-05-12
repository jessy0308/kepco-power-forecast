import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    RandomForestRegressor = None

def train_and_evaluate_ml():
    print("AI 기반 날씨 연계 전력수요 예측 모델(Random Forest) 학습을 시작합니다...\n")
    
    # 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(base_dir, 'forecast', 'data', 'merged_power_weather_2025.csv')
    
    if not os.path.exists(data_file):
        print(f"오류: 병합된 데이터 파일이 없습니다: {data_file}")
        return
        
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 시계열 특성 추출 (시간, 요일, 월 등)
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    print(f"데이터 로드 및 시계열 특성 생성 완료: 총 {len(df)}행")
    
    if RandomForestRegressor is None:
        print("\n[알림] scikit-learn 라이브러리가 설치되어 있지 않습니다.")
        print("터미널에서 다음 명령어를 실행하여 설치해 주세요:")
        print("uv pip install scikit-learn")
        return
        
    # 특성(X)과 타겟(y) 정의
    # 날씨 변수와 시계열 패턴을 모두 활용하여 전력수요 예측
    features = ['temperature', 'humidity', 'wind_speed', 'hour', 'dayofweek', 'month']
    target = 'power_demand'
    
    X = df[features]
    y = df[target]
    
    # 시계열 분할: 마지막 7일(168시간)을 검증용 테스트 셋으로 분리
    test_hours = 7 * 24
    
    X_train = X.iloc[:-test_hours]
    y_train = y.iloc[:-test_hours]
    
    X_test = X.iloc[-test_hours:]
    y_test = y.iloc[-test_hours:]
    test_dates = df['datetime'].iloc[-test_hours:]
    
    print(f"학습 데이터: {len(X_train)}행 / 테스트 데이터: {len(X_test)}행 (마지막 7일)")
    
    # Random Forest 회귀 모델 정의 및 학습
    print("\nRandom Forest 모델 학습 진행 중 (n_estimators=100)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("모델 학습 완료!")
    
    # 예측 수행
    y_pred = model.predict(X_test)
    
    # 성능 평가
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\n[예측 성능 평가 결과]")
    print(f"MAE (평균 절대 오차): {mae:.2f} MW")
    print(f"RMSE (평균 제곱근 오차): {rmse:.2f} MW")
    print(f"MAPE (평균 절대 백분율 오차): {mape:.2f}%")
    
    # 변수 중요도(Feature Importance) 분석
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n[변수 중요도 분석 (어떤 변수가 전력수요에 가장 큰 영향을 미쳤는가?)]")
    for f in range(len(features)):
        print(f"{f+1}. {features[indices[f]]}: {importances[indices[f]]:.4f}")
        
    # 결과 시각화 1: 예측값 vs 실제값 비교 그래프
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test.values, label='Actual Demand', color='black', linewidth=1.5)
    plt.plot(test_dates, y_pred, label='Predicted Demand (RF)', color='crimson', linestyle='--', linewidth=1.5)
    
    plt.title('Power Demand Forecast vs Actual (Machine Learning)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Power Demand (MW)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    images_dir = os.path.join(base_dir, 'forecast', 'images')
    os.makedirs(images_dir, exist_ok=True)
    img_path1 = os.path.join(images_dir, 'ml_forecast_actual_vs_pred.png')
    plt.savefig(img_path1, dpi=300)
    print(f"\n예측 비교 그래프 저장 완료: {os.path.abspath(img_path1)}")
    
    # 결과 시각화 2: 변수 중요도 차트
    plt.figure(figsize=(8, 5))
    sorted_features = [features[i] for i in indices]
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), sorted_features)
    plt.gca().invert_yaxis()  # 상위 항목을 위로
    plt.title('Feature Importances for Power Demand Prediction', fontsize=12, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=10)
    plt.tight_layout()
    
    img_path2 = os.path.join(images_dir, 'ml_feature_importances.png')
    plt.savefig(img_path2, dpi=300)
    print(f"변수 중요도 차트 저장 완료: {os.path.abspath(img_path2)}")
    
    # 예측 상세 결과 CSV 저장
    results_df = pd.DataFrame({
        'datetime': test_dates,
        'actual_demand': y_test,
        'predicted_demand': y_pred,
        'error': np.abs(y_test - y_pred)
    })
    
    report_dir = os.path.join(base_dir, 'forecast', 'report')
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, 'ml_forecast_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"예측 상세 결과 CSV 저장 완료: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    train_and_evaluate_ml()
