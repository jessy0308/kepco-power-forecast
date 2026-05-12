import os
import re

def restore_relative_paths(report_path):
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found")
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 이미지 파일명 리스트 (순서대로 매칭하기 위함)
    image_files = [
        "01_month_freq.png", "02_weekday_freq.png", "03_daily_total_dist.png",
        "04_monthly_avg_demand.png", "05_weekday_boxplot.png", "06_hourly_avg_trend.png",
        "07_hourly_boxplot.png", "08_annual_trend.png", "09_quarter_hourly_trend.png",
        "10_quarter_weekend_compare.png", "11_min_max_scatter.png"
    ]

    # <img ...> 태그를 정규식으로 찾아서 원래의 마크다운 이미지 문법으로 복구
    # <img> 태그 내의 alt 속성을 추출하여 제목으로 사용
    img_tag_pattern = re.compile(r'<img src="data:image/png;base64,[^"]+" alt="([^"]+)" style="[^"]+">')
    
    matches = img_tag_pattern.findall(content)
    
    new_content = content
    for i, alt_text in enumerate(matches):
        if i < len(image_files):
            replacement = f"![{alt_text}](../images/{image_files[i]})"
            # 첫 번째 발견되는 img 태그만 순차적으로 치환
            new_content = img_tag_pattern.sub(replacement, new_content, count=1)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Successfully restored relative paths in {report_path}")

if __name__ == "__main__":
    restore_relative_paths('kepco/report/eda_report.md')
