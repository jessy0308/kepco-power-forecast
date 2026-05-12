# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import os
import io

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', '한국전력거래소_시간별 전국 전력수요량_20251231.csv')
IMG_DIR = os.path.join(BASE_DIR, 'images')
REPORT_PATH = os.path.join(BASE_DIR, 'report', 'eda_report.md')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# Markdown elements
md_content = []
md_content.append("# 한국전력거래소 시간별 전국 전력수요량 EDA 보고서\n")

# 1. 데이터 로드 및 기본 정보
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='cp949')
except Exception as e:
    df = pd.read_csv(DATA_PATH)

# 컬럼명 강제 지정 (파일 인코딩 깨짐 현상 방지)
df.columns = ['날짜'] + [f"{i}시" for i in range(1, 25)]


md_content.append("## 1. 기본 데이터 정보\n")

# 상위 5개행
md_content.append("### 상위 5개 행\n")
md_content.append(df.head(5).to_markdown(index=False) + "\n")

# 하위 5개행
md_content.append("### 하위 5개 행\n")
md_content.append(df.tail(5).to_markdown(index=False) + "\n")

# Info()
buffer = io.StringIO()
df.info(buf=buffer)
md_content.append("### 데이터 정보 (info)\n")
md_content.append("```text\n" + buffer.getvalue() + "```\n")

# Shape & Duplicates
md_content.append(f"- **전체 행의 수**: {df.shape[0]}\n")
md_content.append(f"- **전체 열의 수**: {df.shape[1]}\n")
md_content.append(f"- **중복 데이터 수**: {df.duplicated().sum()}\n\n")

# 2. 데이터 전처리 및 파생 변수 생성
df['날짜'] = pd.to_datetime(df['날짜'])
df['월'] = df['날짜'].dt.month
df['요일'] = df['날짜'].dt.day_name()
# 한국어 요일로 변환
weekday_map = {'Monday': '월요일', 'Tuesday': '화요일', 'Wednesday': '수요일', 'Thursday': '목요일', 'Friday': '금요일', 'Saturday': '토요일', 'Sunday': '일요일'}
df['요일'] = df['요일'].map(weekday_map)
df['분기'] = df['날짜'].dt.quarter.astype(str) + '분기'

hour_cols = [f"{i}시" for i in range(1, 25)]
df['일평균'] = df[hour_cols].mean(axis=1)
df['일최대'] = df[hour_cols].max(axis=1)
df['일최소'] = df[hour_cols].min(axis=1)
df['일합계'] = df[hour_cols].sum(axis=1)

md_content.append("## 2. 기술 통계\n")
md_content.append("### 수치형 데이터 기술 통계\n")
num_cols = hour_cols + ['일평균', '일최대', '일최소', '일합계']
md_content.append(df[num_cols].describe().to_markdown() + "\n")

md_content.append("### 범주형 데이터 빈도수\n")
cat_cols = ['월', '요일', '분기']
for col in cat_cols:
    md_content.append(f"#### {col} 빈도수\n")
    vc = df[col].value_counts().head(30)
    md_content.append(vc.to_frame('빈도수').to_markdown() + "\n")

# 3. 데이터 시각화
# 3. 데이터 시각화 (전문적인 다크 테마 스타일 적용)
plt.rcParams['figure.facecolor'] = '#0f172a'
plt.rcParams['axes.facecolor'] = '#1e293b'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.labelcolor'] = '#cbd5e1'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['text.color'] = '#f8fafc'
plt.rcParams['grid.color'] = '#334155'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5

# 전문적인 컬러 팔레트 정의
PRIMARY_COLOR = '#38bdf8'  # Sky Blue
SECONDARY_COLOR = '#fbbf24' # Amber/Gold
ACCENT_COLOR = '#10b981'   # Emerald
DANGER_COLOR = '#f43f5e'   # Rose

md_content.append("## 3. 데이터 시각화 분석\n")

# Helper function to add visualization
def add_viz(title, img_name, table_md, desc):
    md_content.append(f"### {title}\n")
    md_content.append(f"![{title}](../images/{img_name})\n\n")
    md_content.append(f"**데이터 표:**\n\n{table_md}\n\n")
    md_content.append(f"**해석:**\n{desc}\n\n")

# 1. 월별 관측치 빈도수
plt.figure(figsize=(10, 6))
df['월'].value_counts().sort_index().plot(kind='bar', color=PRIMARY_COLOR, alpha=0.8, edgecolor=PRIMARY_COLOR, linewidth=1)
plt.title('Monthly Observation Frequency', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Days', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_month_freq.png'))
plt.close()

table_md = df['월'].value_counts().sort_index().to_frame('빈도수').to_markdown()
desc = "각 월별로 데이터가 얼마나 수집되었는지 보여주는 빈도수 막대 그래프입니다. 대부분의 달이 30일 혹은 31일로 균일하게 관측되었으며, 결측된 날짜가 거의 없음을 확인할 수 있어 전력수요 분석의 기초 데이터로서의 무결성이 높습니다."
add_viz("1. 월별 관측치 빈도수", "01_month_freq.png", table_md, desc)

# 2. 요일별 관측치 빈도수
plt.figure(figsize=(10, 6))
order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
df['요일'].value_counts().reindex(order).plot(kind='bar', color=ACCENT_COLOR, alpha=0.8, edgecolor=ACCENT_COLOR, linewidth=1)
plt.title('Weekly Observation Frequency', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Days', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_weekday_freq.png'))
plt.close()

table_md = df['요일'].value_counts().reindex(order).to_frame('빈도수').to_markdown()
desc = "요일별 데이터 수집 일수를 나타내는 그래프입니다. 일주일 간의 데이터가 고르게 수집되었으며, 특정한 요일에 데이터가 누락되거나 편중되지 않았음을 알 수 있습니다. 이는 요일별 전력 소비 패턴을 비교할 때 편향되지 않은 결과를 보장합니다."
add_viz("2. 요일별 관측치 빈도수", "02_weekday_freq.png", table_md, desc)

# 3. 일일 총 전력수요량 분포
plt.figure(figsize=(10, 6))
plt.hist(df['일합계'], bins=30, color=SECONDARY_COLOR, alpha=0.7, edgecolor='#0f172a', linewidth=0.5)
plt.title('Daily Total Demand Distribution', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Total Demand (MWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '03_daily_total_dist.png'))
plt.close()

table_md = df['일합계'].describe().to_frame('일일 총 전력수요량 통계').to_markdown()
desc = "일일 총 전력수요량의 전체적인 분포를 보여주는 히스토그램입니다. 데이터가 대략적으로 정규분포와 유사한 형태를 띠고 있지만, 꼬리 부분이 한쪽으로 치우쳐 있는 양상을 보입니다. 이는 특정한 기상 조건(폭염, 한파)에 의해 극단적으로 전력수요가 급증하는 날이 존재함을 암시합니다."
add_viz("3. 일일 총 전력수요량 분포 (히스토그램)", "03_daily_total_dist.png", table_md, desc)

# 4. 월별 평균 전력수요량 (Bar)
plt.figure(figsize=(10, 6))
monthly_avg = df.groupby('월')['일합계'].mean()
monthly_avg.plot(kind='bar', color=PRIMARY_COLOR, alpha=0.9, edgecolor=PRIMARY_COLOR, linewidth=1)
plt.title('Monthly Average Total Demand', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Demand (MWh)', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '04_monthly_avg_demand.png'))
plt.close()

table_md = monthly_avg.to_frame('월별 평균 전력수요량').to_markdown()
desc = "1월부터 12월까지 월별 평균 전력수요량을 나타내는 막대 그래프입니다. 냉방 수요가 많은 7~8월과 난방 수요가 많은 12~2월에 수요가 크게 증가하는 전형적인 계절적 패턴(W자 혹은 U자형)을 뚜렷하게 확인할 수 있습니다. 봄, 가을철은 상대적으로 수요가 적습니다."
add_viz("4. 월별 평균 총 전력수요량", "04_monthly_avg_demand.png", table_md, desc)

# 5. 요일별 일평균 전력수요량 (Boxplot)
plt.figure(figsize=(10, 6))
boxplot_data = [df[df['요일'] == day]['일평균'] for day in order]
box = plt.boxplot(boxplot_data, labels=order, patch_artist=True)
for patch in box['boxes']:
    patch.set_facecolor('#334155')
    patch.set_edgecolor(PRIMARY_COLOR)
for median in box['medians']:
    median.set_color(SECONDARY_COLOR)
plt.title('Weekly Power Demand Distribution', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Average Demand (MWh)', fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '05_weekday_boxplot.png'))
plt.close()

table_md = df.groupby('요일')['일평균'].describe().reindex(order).to_markdown()
desc = "요일별 전력수요량의 분포를 보여주는 상자 수염 그림(Boxplot)입니다. 평일(월~금)에는 산업 및 상업용 전력 사용으로 인해 평균 수요가 높고 중앙값이 일관되게 유지되나, 주말(토, 일)에는 상업 시설 휴무 등으로 인해 전력 수요가 뚜렷하게 하락하는 경향을 보여줍니다."
add_viz("5. 요일별 평균 전력수요량 분포", "05_weekday_boxplot.png", table_md, desc)

# 6. 시간대별(1시~24시) 평균 전력수요량 추이 (Line)
plt.figure(figsize=(12, 6))
hourly_avg = df[hour_cols].mean()
plt.plot(range(1, 25), hourly_avg, marker='o', markersize=6, linewidth=2.5, color=PRIMARY_COLOR, markerfacecolor=SECONDARY_COLOR, markeredgecolor=SECONDARY_COLOR)
plt.title('Hourly Average Demand Trend', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Average Demand (MWh)', fontsize=12)
plt.xticks(range(1, 25))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '06_hourly_avg_trend.png'))
plt.close()

table_md = hourly_avg.to_frame('시간대별 평균').to_markdown()
desc = "하루 24시간 동안의 평균 전력수요량 변화 추이를 나타냅니다. 심야 시간대(새벽 3~5시)에 수요가 가장 낮으며, 아침 출근 및 일과 시작 시간에 급격히 상승하여 오후 시간에 정점을 찍고, 퇴근 이후 밤 시간에 서서히 감소하는 전형적인 일간 라이프사이클을 보여주고 있습니다."
add_viz("6. 전체 시간대별 평균 전력수요량 추이", "06_hourly_avg_trend.png", table_md, desc)

# 7. 시간대별 전력수요량 분포 (Boxplot)
plt.figure(figsize=(14, 6))
# 데이터를 리스트 형태로 명시적 변환하여 박스 누락 방지
boxplot_hourly_data = [df[col].dropna().values for col in hour_cols]
box = plt.boxplot(boxplot_hourly_data, tick_labels=[f"{i}" for i in range(1, 25)], patch_artist=True)

for patch in box['boxes']:
    patch.set_facecolor('#1e293b')
    patch.set_edgecolor(ACCENT_COLOR)
    patch.set_alpha(0.8)
for median in box['medians']:
    median.set_color(SECONDARY_COLOR)
    median.set_linewidth(2)
for whisker in box['whiskers']:
    whisker.set_color('#475569')
for cap in box['caps']:
    cap.set_color('#475569')

plt.title('Hourly Power Demand Distribution', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Demand (MWh)', fontsize=12)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '07_hourly_boxplot.png'))
plt.close()

table_md = df[hour_cols].describe().T.to_markdown()
desc = "24시간 각각의 전력수요량에 대한 분포 범위를 보여주는 박스플롯입니다. 활동이 많은 주간 시간대(특히 오후)에는 데이터의 변동 폭(IQR)이 넓고 이상치가 다수 발생하여 계절이나 요일에 따른 변동성이 큼을 의미하며, 심야 시간대에는 상대적으로 변동 폭이 작음을 알 수 있습니다."
add_viz("7. 전체 시간대별 전력수요량 분포", "07_hourly_boxplot.png", table_md, desc)

# 8. 날짜(시간흐름)에 따른 일일 총 전력수요량 추이
plt.figure(figsize=(14, 6))
plt.plot(df['날짜'], df['일합계'], linestyle='-', linewidth=1.5, color=PRIMARY_COLOR, alpha=0.8)
plt.title('Annual Daily Total Demand Trend', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Demand (MWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_annual_trend.png'))
plt.close()

table_md = df.groupby('월')['일합계'].agg(['mean', 'max', 'min']).to_markdown()
desc = "연간 일일 단위의 전력수요량 흐름을 시계열로 나타낸 그래프입니다. 여름과 겨울철에 피크가 확연히 나타나고 있으며, 각 주마다 평일과 주말의 수요 차이로 인한 미세한 톱니바퀴 모양의 주기성(Seasonality)이 반복되는 현상을 관찰할 수 있습니다. 피크 타임 관리가 중요함을 시사합니다."
add_viz("8. 연간 일일 총 전력수요량 추이", "08_annual_trend.png", table_md, desc)

# 9. 계절(분기)별 시간대별 전력수요량 평균 비교 (Line)
plt.figure(figsize=(12, 6))
quarter_hourly = df.groupby('분기')[hour_cols].mean()
colors = [PRIMARY_COLOR, ACCENT_COLOR, SECONDARY_COLOR, DANGER_COLOR]
for (idx, row), color in zip(quarter_hourly.iterrows(), colors):
    plt.plot(range(1, 25), row, marker='o', markersize=4, label=idx, color=color, linewidth=2)
plt.title('Hourly Demand Trend by Quarter', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Average Demand (MWh)', fontsize=12)
plt.xticks(range(1, 25))
plt.legend(facecolor='#1e293b', edgecolor='#334155')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '09_quarter_hourly_trend.png'))
plt.close()

table_md = quarter_hourly.T.to_markdown()
desc = "사계절을 대표하는 각 분기별로 24시간 전력수요 패턴을 겹쳐 그린 다중 선 그래프입니다. 난방 수요가 큰 1분기(겨울)와 냉방 수요가 집중되는 3분기(여름)의 전체 레벨이 높게 형성되며, 2분기와 4분기는 상대적으로 낮게 나타나 계절에 따라 하루 전력 사용의 기준선이 다름을 잘 보여줍니다."
add_viz("9. 분기별 시간대별 전력수요량 추이 비교", "09_quarter_hourly_trend.png", table_md, desc)

# 10. 분기별 평일/주말 평균 전력수요량 비교 (Bar)
df['주말여부'] = df['요일'].apply(lambda x: '주말' if x in ['토요일', '일요일'] else '평일')
weekend_quarter = df.pivot_table(index='분기', columns='주말여부', values='일합계', aggfunc='mean')

plt.figure(figsize=(10, 6))
weekend_quarter.plot(kind='bar', color=[PRIMARY_COLOR, DANGER_COLOR], alpha=0.8, ax=plt.gca(), edgecolor='#0f172a', linewidth=0.5)
plt.title('Weekday vs Weekend Comparison by Quarter', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Average Demand (MWh)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Category', facecolor='#1e293b', edgecolor='#334155')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '10_quarter_weekend_compare.png'))
plt.close()

table_md = weekend_quarter.to_markdown()
desc = "각 분기마다 평일과 주말의 평균 전력 수요를 대비하여 보여주는 교차 막대그래프입니다. 모든 분기에서 예외 없이 평일 전력 수요가 주말보다 압도적으로 높게 나타나며, 특히 1분기와 3분기의 평일 피크가 가장 높습니다. 이는 상업/산업용 전력이 전체 수요에 미치는 영향력이 매우 지대함을 시사합니다."
add_viz("10. 분기별 평일 vs 주말 평균 전력수요량 비교", "10_quarter_weekend_compare.png", table_md, desc)

# 11. 최대/최소 수요량 간의 산점도
plt.figure(figsize=(8, 8))
plt.scatter(df['일최소'], df['일최대'], alpha=0.5, color=PRIMARY_COLOR, s=40, edgecolors='#1e293b', linewidth=0.5)
plt.title('Daily Min vs Max Demand Correlation', fontsize=15, pad=20, fontweight='bold')
plt.xlabel('Min Demand (MWh)', fontsize=12)
plt.ylabel('Max Demand (MWh)', fontsize=12)
plt.grid(True, alpha=0.2)
# 추세선 추가 (수치 안정성을 위해 단순 선형 회귀 적용)
try:
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['일최소'], df['일최대'])
    line = slope * df['일최소'] + intercept
    plt.plot(df['일최소'], line, color=DANGER_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
except:
    pass # scipy 미설치 시 추세선 생략

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '11_min_max_scatter.png'))
plt.close()

table_md = df[['일최소', '일최대']].describe().to_markdown()
desc = "하루 동안 기록된 가장 적은 수요량(일최소)과 가장 많은 수요량(일최대)의 상관관계를 보여주는 산점도입니다. 최소 전력수요량이 높은 날일수록 최대 전력수요량도 정비례하여 증가하는 강한 양의 상관관계를 보여줍니다. 점들의 산포가 넓게 퍼져있는 부분은 일교차가 크거나 갑작스러운 기온 변화가 있던 날로 추정됩니다."
add_viz("11. 일최소 vs 일최대 전력수요량 산점도", "11_min_max_scatter.png", table_md, desc)



# Write report
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(md_content))

print(f"EDA Report successfully generated at: {REPORT_PATH}")
