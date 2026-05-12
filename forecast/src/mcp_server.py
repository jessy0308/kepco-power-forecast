import os
import json
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
import httpx
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

try:
    from prophet.serialize import model_from_json
except ImportError:
    model_from_json = None

from dotenv import load_dotenv

# 환경변수 로드 (.env 경로 맞춤)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(base_dir, 'weather', '.env')
load_dotenv(env_path)

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
FORECAST_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

# FastMCP 서버 초기화
mcp = FastMCP("kepco_forecast_mcp")

# 입력 스키마 정의
class CurrentWeatherInput(BaseModel):
    '''기상청 실시간 실황 조회를 위한 입력 파라미터'''
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    nx: int = Field(default=60, description="X 격자 좌표 (예: 서울은 60)", ge=1, le=149)
    ny: int = Field(default=127, description="Y 격자 좌표 (예: 서울은 127)", ge=1, le=253)

class TomorrowDemandInput(BaseModel):
    '''내일 예상 전력수요량 조회를 위한 입력 파라미터'''
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    nx: int = Field(default=60, description="예보를 참조할 X 격자 좌표", ge=1, le=149)
    ny: int = Field(default=127, description="예보를 참조할 Y 격자 좌표", ge=1, le=253)


# 공통 유틸리티
def get_ultrasrt_base_datetime() -> tuple[str, str]:
    now = datetime.now()
    if now.minute < 40:
        now = now - timedelta(hours=1)
    return now.strftime('%Y%m%d'), now.strftime('%H00')

def get_vilage_base_datetime() -> tuple[str, str, str]:
    now = datetime.now()
    today = now.strftime('%Y%m%d')
    tomorrow = (now + timedelta(days=1)).strftime('%Y%m%d')
    
    # 단기예보 제공 시간: 02:00, 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00 (각 10분 후 발표)
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
    return base_date, base_time, tomorrow

# 도구: 기상 실황 조회
@mcp.tool(
    name="get_current_weather",
    annotations={
        "title": "기상청 실시간 실황 조회",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True
    }
)
async def get_current_weather(params: CurrentWeatherInput) -> str:
    '''현재 시점의 기상청 초단기실황 데이터를 JSON 형태로 반환합니다.

    Args:
        params (CurrentWeatherInput): 격자 좌표(nx, ny)

    Returns:
        str: 수집된 주요 날씨 데이터 (기온, 습도, 풍속 등)
    '''
    if not WEATHER_API_KEY:
        return json.dumps({"error": "WEATHER_API_KEY가 설정되지 않았습니다."})

    base_date, base_time = get_ultrasrt_base_datetime()
    
    # API 키 디코딩 처리 (필요시)
    decoded_key = urllib.parse.unquote(WEATHER_API_KEY) if '%' in WEATHER_API_KEY else WEATHER_API_KEY

    request_params = {
        'serviceKey': decoded_key,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': str(params.nx),
        'ny': str(params.ny)
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(WEATHER_URL, params=request_params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            header = data.get('response', {}).get('header', {})
            if header.get('resultCode') != '00':
                return json.dumps({"error": f"API 호출 실패: {header.get('resultMsg')}"})

            items = data['response']['body']['items']['item']
            
            category_names = {
                'T1H': '기온(℃)',
                'RN1': '1시간 강수량(mm)',
                'REH': '습도(%)',
                'PTY': '강수형태',
                'VEC': '풍향(deg)',
                'WSD': '풍속(m/s)'
            }
            
            result = {"base_date": base_date, "base_time": base_time, "nx": params.nx, "ny": params.ny, "weather": {}}
            for item in items:
                cat = item['category']
                if cat in category_names:
                    try:
                        result["weather"][category_names[cat]] = float(item['obsrValue'])
                    except ValueError:
                        result["weather"][category_names[cat]] = item['obsrValue']

            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"네트워크 또는 처리 오류 발생: {str(e)}"})


# 도구: 내일 전력수요 예측
@mcp.tool(
    name="predict_tomorrow_power_demand",
    annotations={
        "title": "내일 기상 예보 기반 전력수요 예측",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True
    }
)
async def predict_tomorrow_power_demand(params: TomorrowDemandInput) -> str:
    '''기상청 단기예보를 조회하여 내일의 기온, 습도, 풍속을 수집한 뒤, 
    미리 학습된 AI(Prophet) 모델을 이용해 시간별 전력수요량을 예측하여 결과를 반환합니다.

    Args:
        params (TomorrowDemandInput): 예보를 참조할 격자 좌표(nx, ny)

    Returns:
        str: 시간별 날씨 예보 및 전력수요량 예측값 JSON
    '''
    if not WEATHER_API_KEY:
        return json.dumps({"error": "WEATHER_API_KEY가 설정되지 않았습니다."})
    
    if model_from_json is None:
        return json.dumps({"error": "Prophet 모듈을 사용할 수 없습니다."})

    # 1. 기상청 단기예보 수집
    base_date, base_time, tomorrow_date = get_vilage_base_datetime()
    decoded_key = urllib.parse.unquote(WEATHER_API_KEY) if '%' in WEATHER_API_KEY else WEATHER_API_KEY

    request_params = {
        'serviceKey': decoded_key,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': str(params.nx),
        'ny': str(params.ny)
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(FORECAST_URL, params=request_params, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            items = data['response']['body']['items']['item']
        except Exception as e:
            return json.dumps({"error": f"단기예보 수집 중 오류: {str(e)}"})

    # 예보 데이터 정리 (내일 데이터만 필터링)
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
        return json.dumps({"error": "내일의 기상 예보 데이터를 파싱할 수 없습니다."})

    df_future['ds'] = pd.to_datetime(df_future['ds'])
    df_future = df_future.sort_values('ds')

    # 2. Prophet 모델 로드 및 예측 수행
    model_path = os.path.join(base_dir, 'forecast', 'models', 'prophet_model.json')
    if not os.path.exists(model_path):
        return json.dumps({"error": f"학습된 예측 모델 파일을 찾을 수 없습니다: {model_path}"})
        
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(fin.read())
            
        forecast = model.predict(df_future)
        
        # 3. 예측 결과 결합 및 포맷팅
        df_result = df_future.copy()
        df_result['predicted_demand_MW'] = forecast['yhat'].values.round(2)
        df_result['ds'] = df_result['ds'].dt.strftime('%Y-%m-%d %H:%M')
        
        output_data = {
            "target_date": tomorrow_date,
            "predictions": df_result.to_dict(orient='records')
        }
        return json.dumps(output_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
         return json.dumps({"error": f"예측 모델 구동 중 오류: {str(e)}"})

if __name__ == "__main__":
    # MCP 서버 구동 (stdio 전송)
    mcp.run()
