import base64
import os

def get_b64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""

print("IMG04:" + get_b64("kepco/images/04_monthly_avg_demand.png"))
print("IMG06:" + get_b64("kepco/images/06_hourly_avg_trend.png"))
print("IMG09:" + get_b64("kepco/images/09_quarter_hourly_trend.png"))
