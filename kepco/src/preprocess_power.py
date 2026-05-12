import pandas as pd
import os

def preprocess_power_data(input_path, output_path):
    print(f"데이터 불러오는 중: {input_path}")
    # 한국 공공데이터의 일반적인 인코딩인 cp949 사용
    df = pd.read_csv(input_path, encoding='cp949')
    
    print(f"원본 데이터 형태: {df.shape}")
    
    # 1. 합계나 평균 등 결측치가 있는 하단 집계 행 제거
    df = df.dropna(subset=[df.columns[0]])
    
    date_col = df.columns[0]  # 보통 '날짜'
    hour_cols = df.columns[1:] # '1시', '2시' ... '24시'
    
    # 2. Wide 포맷(가로)을 Long 포맷(세로)으로 변환 (Melt)
    df_melted = df.melt(id_vars=[date_col], value_vars=hour_cols, 
                        var_name='hour_str', value_name='power_demand')
    
    # 3. 시간 변환 ('1시' -> 정수 1 추출)
    df_melted['hour'] = df_melted['hour_str'].astype(str).str.extract(r'(\d+)').astype(int)
    
    # 날짜를 datetime 타입으로 변환
    df_melted['date'] = pd.to_datetime(df_melted[date_col])
    
    # 한국전력 데이터의 '24시'는 자정(다음날 0시)을 의미하므로 이를 처리
    mask_24 = df_melted['hour'] == 24
    df_melted.loc[mask_24, 'date'] = df_melted.loc[mask_24, 'date'] + pd.Timedelta(days=1)
    df_melted.loc[mask_24, 'hour'] = 0
    
    # 4. 하나의 'datetime' 컬럼으로 통합
    df_melted['datetime'] = df_melted['date'] + pd.to_timedelta(df_melted['hour'], unit='h')
    
    # 5. 불필요한 컬럼 제거 및 정렬
    df_clean = df_melted[['datetime', 'power_demand']].sort_values('datetime').reset_index(drop=True)
    
    # 결측치 확인
    null_counts = df_clean.isnull().sum()
    if null_counts['power_demand'] > 0:
        print(f"경고: 전력수요 데이터에 결측치가 {null_counts['power_demand']}개 있습니다. 보간(Interpolation)을 수행합니다.")
        df_clean['power_demand'] = df_clean['power_demand'].interpolate(method='linear')
    
    print(f"전처리 완료 데이터 형태: {df_clean.shape}")
    print("\n--- 전처리 결과 미리보기 (상위 5행) ---")
    print(df_clean.head())
    
    # 6. CSV로 저장
    df_clean.to_csv(output_path, index=False)
    print(f"\n성공적으로 저장되었습니다: {output_path}")

if __name__ == "__main__":
    # 절대 경로 계산
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'data', '한국전력거래소_시간별 전국 전력수요량_20251231.csv')
    output_file = os.path.join(base_dir, 'data', 'power_demand_hourly.csv')
    
    preprocess_power_data(input_file, output_file)
