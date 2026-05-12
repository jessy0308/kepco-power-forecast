import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

API_KEY = os.getenv('KMA_APIHUB_KEY')

def fetch_apihub_data(tm1='2025010100', tm2='2025123123', stn='108'):
    """
    기상청 API 허브를 통해 과거 종관기상관측(ASOS) 시간단위 데이터를 기간으로 일괄 수집합니다.
    tm1: 시작 일시 (YYYYMMDDHH)
    tm2: 종료 일시 (YYYYMMDDHH)
    stn: 지점 번호 (108: 서울)
    """
    if not API_KEY or API_KEY == "여기에_기상청_API허브_인증키를_입력하세요":
        print("오류: KMA_APIHUB_KEY가 정상적으로 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return None
        
    url = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php'
    params = {
        'tm1': tm1,
        'tm2': tm2,
        'stn': stn,
        'help': '1',  # 컬럼명 등 안내 텍스트 포함
        'authKey': API_KEY
    }
    
    print(f"API 허브 데이터 수집 요청: {tm1} ~ {tm2} (지점: {stn})...")
    response = requests.get(url, params=params, timeout=60)
    
    if response.status_code != 200:
        print(f"API 요청 실패: HTTP {response.status_code}")
        print(f"응답 내용: {response.text[:500]}")
        return None
        
    # 한글 인코딩 처리 (보통 EUC-KR 또는 UTF-8 사용)
    response.encoding = 'euc-kr' if 'euc-kr' in response.headers.get('content-type', '').lower() else 'utf-8'
    text_data = response.text
    
    # 인증 실패나 에러 메시지가 있는지 확인
    if "AUTH_KEY" in text_data or "오류" in text_data or "error" in text_data.lower():
        # 정상 데이터의 경우 앞부분에 주석(#)이나 날짜가 포함됨
        if not any(char.isdigit() for char in text_data[:100]):
            print("API 응답에 오류 메시지가 포함되어 있을 수 있습니다:")
            print(text_data[:500])
            return None
            
    return text_data

def process_and_save(text_data, output_filename='weather_apihub_2025.csv'):
    """
    API 허브에서 수신한 텍스트 데이터를 파싱하여 데이터프레임으로 변환하고 CSV로 저장합니다.
    """
    if not text_data:
        return
        
    lines = text_data.split('\n')
    print(f"총 {len(lines)} 줄의 응답 수신 완료. 데이터 파싱 시작...")
    
    # 주석(#)으로 시작하지 않는 실제 데이터 행만 추출
    data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    # 컬럼 헤더 찾기 (보통 데이터 바로 앞의 # 행)
    header_line = None
    for line in lines:
        if line.strip().startswith('#'):
            if 'YYMMDDHHMI' in line or 'STN' in line or 'TM' in line:
                header_line = line.replace('#', '').strip()
                
    if not data_lines:
        print("추출할 데이터 행이 없습니다. API 응답 텍스트 확인:")
        print('\n'.join(lines[:20]))
        return
        
    # 공백을 기준으로 분리하여 2차원 리스트 생성
    rows = [line.split() for line in data_lines]
    num_cols = len(rows[0])
    print(f"데이터 컬럼 수: {num_cols}개 감지됨")
    
    # 컬럼명 매핑
    if header_line:
        cols = header_line.split()
        if len(cols) != num_cols:
            print(f"헤더 컬럼 수({len(cols)})와 데이터 컬럼 수({num_cols})가 달라 임의 컬럼명을 부여합니다.")
            cols = [f"col_{i}" for i in range(num_cols)]
    else:
        cols = [f"col_{i}" for i in range(num_cols)]
        
    df = pd.DataFrame(rows, columns=cols)
    print("\n[파싱된 원본 데이터 5행]")
    print(df.head())
    
    # 필수 타겟 컬럼 찾기 (대소문자 구분 없이 매핑)
    col_map = {}
    for c in df.columns:
        c_upper = c.upper()
        if 'YYMMDDHHMI' in c_upper or 'TM' in c_upper:
            if 'datetime' not in col_map.values():
                col_map[c] = 'datetime'
        elif c_upper == 'TA':
            col_map[c] = 'temperature'
        elif c_upper == 'RN':
            col_map[c] = 'rainfall'
        elif c_upper == 'WS':
            col_map[c] = 'wind_speed'
        elif c_upper == 'HM':
            col_map[c] = 'humidity'
            
    print(f"\n매핑된 컬럼 정보: {col_map}")
    
    # 추출된 컬럼만 모아 새로운 데이터프레임 구성
    df_clean = pd.DataFrame()
    for orig_col, new_col in col_map.items():
        df_clean[new_col] = df[orig_col].copy()
        
    # 누락된 기본 컬럼이 있다면 빈 값으로 생성
    for required in ['datetime', 'temperature', 'rainfall', 'wind_speed', 'humidity']:
        if required not in df_clean.columns:
            print(f"경고: '{required}' 컬럼을 응답에서 찾지 못했습니다.")
            df_clean[required] = pd.NA
            
    # 날짜/시간 포맷 변환 (예: 202501010000 -> 2025-01-01 00:00:00)
    if not df_clean['datetime'].isna().all():
        df_clean['datetime'] = df_clean['datetime'].astype(str)
        # 시간 단위 매핑을 위해 앞에서부터 10자리(YYYYMMDDHH)만 사용
        df_clean['datetime'] = df_clean['datetime'].apply(lambda x: x[:10] if len(x) >= 10 else x)
        df_clean['datetime'] = pd.to_datetime(df_clean['datetime'], format='%Y%m%d%H', errors='coerce')
        
    # 숫자형 변환 및 결측치(-99.0 등) 처리
    for c in df_clean.columns:
        if c != 'datetime':
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
            # API 허브 규격상 -90 이하는 결측치 기호(-99.0 등)
            df_clean.loc[df_clean[c] < -90, c] = pd.NA
            
    # 강수량 결측치는 비가 오지 않은 것(0)으로 처리
    df_clean['rainfall'] = df_clean['rainfall'].fillna(0)
    
    # 기온, 습도 등 일시적 결측치는 앞뒤 데이터를 기반으로 선형 보간
    df_clean = df_clean.interpolate(method='linear')
    
    # 최종 결과 저장
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, output_filename)
    
    df_clean.to_csv(out_path, index=False)
    print(f"\n데이터 전처리 및 CSV 저장 완료! (총 {len(df_clean)}행)")
    print(f"저장 경로: {os.path.abspath(out_path)}")
    print("\n[최종 전처리된 기상 데이터 상위 5행]")
    print(df_clean.head())

if __name__ == "__main__":
    raw_text = fetch_apihub_data('2025010100', '2025123123', '108')
    if raw_text:
        process_and_save(raw_text)
