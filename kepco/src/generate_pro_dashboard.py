import os
import base64

def get_b64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

IMG_DIR = "images"
OUT_FILE = "report/eda_dashboard_preview.html"

images = {
    "01": get_b64(f"{IMG_DIR}/01_month_freq.png"),
    "02": get_b64(f"{IMG_DIR}/02_weekday_freq.png"),
    "03": get_b64(f"{IMG_DIR}/03_daily_total_dist.png"),
    "04": get_b64(f"{IMG_DIR}/04_monthly_avg_demand.png"),
    "05": get_b64(f"{IMG_DIR}/05_weekday_boxplot.png"),
    "06": get_b64(f"{IMG_DIR}/06_hourly_avg_trend.png"),
    "07": get_b64(f"{IMG_DIR}/07_hourly_boxplot.png"),
    "08": get_b64(f"{IMG_DIR}/08_annual_trend.png"),
    "09": get_b64(f"{IMG_DIR}/09_quarter_hourly_trend.png"),
    "10": get_b64(f"{IMG_DIR}/10_quarter_weekend_compare.png"),
    "11": get_b64(f"{IMG_DIR}/11_min_max_scatter.png"),
}

html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KEPCO Energy Intelligence Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@100;400;700;900&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Pretendard', sans-serif; background-color: #020617; color: #f8fafc; }}
        .glass {{ background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); }}
        .card-hover {{ transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }}
        .card-hover:hover {{ transform: translateY(-8px) scale(1.01); background: rgba(30, 41, 59, 0.6); box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5), 0 0 20px 0 rgba(56, 189, 248, 0.1); }}
        .text-gradient {{ background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .glow-border {{ position: relative; overflow: hidden; }}
        .glow-border::after {{ content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: conic-gradient(from 180deg at 50% 50%, transparent 0deg, #38bdf8 45deg, transparent 90deg); animation: rotate 10s linear infinite; opacity: 0.1; }}
        @keyframes rotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body class="p-6 md:p-12 overflow-x-hidden">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row items-start md:items-center justify-between mb-16 gap-6">
            <div>
                <div class="flex items-center gap-2 mb-2">
                    <span class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
                    <span class="text-xs font-bold tracking-widest text-slate-400 uppercase">Live Intelligence Report</span>
                </div>
                <h1 class="text-5xl md:text-7xl font-black text-gradient leading-tight">KEPCO ENERGY<br>DATA ANALYSIS</h1>
                <p class="text-slate-400 mt-4 max-w-xl text-lg font-light leading-relaxed">전국 시간별 전력수요량 데이터를 분석하여 도출된 에너지 소비 패턴 및 예측 인사이트 대시보드입니다.</p>
            </div>
            <div class="glass p-6 rounded-3xl flex flex-col items-end">
                <div class="text-right">
                    <p class="text-xs text-slate-500 uppercase tracking-widest mb-1">Analysis Date</p>
                    <p class="text-xl font-bold">2026.05.07</p>
                </div>
                <div class="mt-4 flex gap-2">
                    <span class="px-3 py-1 bg-slate-800 rounded-full text-xs text-slate-300">V 2.1 Premium</span>
                    <span class="px-3 py-1 bg-sky-900/30 text-sky-400 rounded-full text-xs font-bold border border-sky-800">Secure Data</span>
                </div>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-16">
            <div class="glass p-8 rounded-3xl glow-border">
                <p class="text-slate-400 text-sm font-medium mb-1">전체 평균 수요</p>
                <p class="text-4xl font-black">63.4 <span class="text-lg text-sky-500 font-normal ml-1">GW</span></p>
                <div class="mt-4 h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div class="h-full bg-sky-500 w-3/4"></div>
                </div>
            </div>
            <div class="glass p-8 rounded-3xl">
                <p class="text-slate-400 text-sm font-medium mb-1">연간 최대 피크</p>
                <p class="text-4xl font-black text-rose-500">92.7 <span class="text-lg text-slate-400 font-normal ml-1">GW</span></p>
                <p class="text-xs text-slate-500 mt-2">August Peak Measured</p>
            </div>
            <div class="glass p-8 rounded-3xl">
                <p class="text-slate-400 text-sm font-medium mb-1">연간 최저 소비</p>
                <p class="text-4xl font-black text-emerald-500">41.2 <span class="text-lg text-slate-400 font-normal ml-1">GW</span></p>
                <p class="text-xs text-slate-500 mt-2">May Minimum Measured</p>
            </div>
            <div class="glass p-8 rounded-3xl">
                <p class="text-slate-400 text-sm font-medium mb-1">데이터 무결성</p>
                <p class="text-4xl font-black">100<span class="text-lg text-slate-400 font-normal ml-1">%</span></p>
                <p class="text-xs text-emerald-500/70 mt-2">No Missing Values</p>
            </div>
        </div>

        <!-- Main Gallery -->
        <section>
            <div class="flex items-center justify-between mb-10">
                <h2 class="text-3xl font-bold tracking-tight">Advanced Analytics Gallery</h2>
                <a href="eda_report.md" class="px-8 py-3 bg-white text-black hover:bg-sky-400 hover:text-white rounded-full text-sm font-black transition-all duration-300 transform active:scale-95 shadow-xl shadow-sky-500/10">FULL REPORT VIEW</a>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-10">
                <!-- Card 1 -->
                <div class="glass rounded-[40px] overflow-hidden card-hover">
                    <div class="p-8 border-b border-white/5 bg-gradient-to-br from-slate-800/50 to-transparent">
                        <h3 class="text-xl font-bold mb-2">시간대별 평균 전력수요 트렌드</h3>
                        <p class="text-slate-400 text-sm font-light">하루 24시간의 전형적인 전력 소비 패턴을 시각화하여 전력망 관리 지표를 제공합니다.</p>
                    </div>
                    <div class="p-4 flex items-center justify-center bg-slate-900/50">
                        <img src="data:image/png;base64,{images['06']}" class="w-full h-auto rounded-2xl" alt="Trend">
                    </div>
                </div>

                <!-- Card 2 -->
                <div class="glass rounded-[40px] overflow-hidden card-hover">
                    <div class="p-8 border-b border-white/5 bg-gradient-to-br from-slate-800/50 to-transparent">
                        <h3 class="text-xl font-bold mb-2">분기별 전력 소비 패턴 비교</h3>
                        <p class="text-slate-400 text-sm font-light">냉/난방 기온 변화에 따른 계절적 수요 변동과 기준선 변화를 정밀 분석합니다.</p>
                    </div>
                    <div class="p-4 flex items-center justify-center bg-slate-900/50">
                        <img src="data:image/png;base64,{images['09']}" class="w-full h-auto rounded-2xl" alt="Trend">
                    </div>
                </div>

                <!-- Card 3 -->
                <div class="glass rounded-[40px] overflow-hidden card-hover">
                    <div class="p-8 border-b border-white/5 bg-gradient-to-br from-slate-800/50 to-transparent">
                        <h3 class="text-xl font-bold mb-2">평일 vs 주말 소비량 대조</h3>
                        <p class="text-slate-400 text-sm font-light">산업 활동 여부에 따른 주간 전력 수요 격차를 분기별로 상세 대조한 데이터입니다.</p>
                    </div>
                    <div class="p-4 flex items-center justify-center bg-slate-900/50">
                        <img src="data:image/png;base64,{images['10']}" class="w-full h-auto rounded-2xl" alt="Comparison">
                    </div>
                </div>

                <!-- Card 4 -->
                <div class="glass rounded-[40px] overflow-hidden card-hover">
                    <div class="p-8 border-b border-white/5 bg-gradient-to-br from-slate-800/50 to-transparent">
                        <h3 class="text-xl font-bold mb-2">피크 상관관계 매트릭스</h3>
                        <p class="text-slate-400 text-sm font-light">일일 최소 및 최대 수요 간의 상관계수를 산출하여 전력 예비율 예측의 근거를 마련합니다.</p>
                    </div>
                    <div class="p-4 flex items-center justify-center bg-slate-900/50">
                        <img src="data:image/png;base64,{images['11']}" class="w-full h-auto rounded-2xl" alt="Correlation">
                    </div>
                </div>
            </div>

            <!-- More Insight Grid -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mt-10">
                <div class="glass p-6 rounded-3xl card-hover flex flex-col items-center">
                    <h4 class="text-slate-400 text-sm mb-4">연간 변동 추이</h4>
                    <img src="data:image/png;base64,{images['08']}" class="w-full rounded-xl" alt="Annual">
                </div>
                <div class="glass p-6 rounded-3xl card-hover flex flex-col items-center">
                    <h4 class="text-slate-400 text-sm mb-4">요일별 분포 범위</h4>
                    <img src="data:image/png;base64,{images['05']}" class="w-full rounded-xl" alt="Boxplot">
                </div>
                <div class="glass p-6 rounded-3xl card-hover flex flex-col items-center">
                    <h4 class="text-slate-400 text-sm mb-4">시간대별 변동폭</h4>
                    <img src="data:image/png;base64,{images['07']}" class="w-full rounded-xl" alt="Hourly Box">
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer class="mt-24 pb-12 border-t border-white/5 pt-8 flex flex-col md:flex-row justify-between items-center gap-6">
            <div class="flex items-center gap-4">
                <span class="text-2xl font-black text-gradient">ENERGY IQ</span>
                <span class="text-slate-600">|</span>
                <p class="text-slate-500 text-sm">© 2026 Intelligence Data Lab. All rights reserved.</p>
            </div>
            <div class="flex gap-6 text-slate-500 text-xs tracking-widest uppercase">
                <a href="#" class="hover:text-sky-400 transition">Privacy</a>
                <a href="#" class="hover:text-sky-400 transition">Terms</a>
                <a href="#" class="hover:text-sky-400 transition">Contact</a>
            </div>
        </footer>
    </div>
</body>
</html>
"""

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Professional Dashboard generated at: {OUT_FILE}")
