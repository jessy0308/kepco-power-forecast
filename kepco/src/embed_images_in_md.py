import os
import base64

def get_b64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ''

report_path = 'kepco/report/eda_report.md'
if not os.path.exists(report_path):
    print(f"Error: {report_path} not found")
    exit(1)

with open(report_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('![') and '../images/' in line:
        start = line.find('../images/') + len('../images/')
        end = line.find(')', start)
        filename = line[start:end]
        path = os.path.join('kepco', 'images', filename)
        b64 = get_b64(path)
        if b64:
            title_start = line.find('[') + 1
            title_end = line.find(']')
            title = line[title_start:title_end]
            new_line = f'<img src="data:image/png;base64,{b64}" alt="{title}" style="max-width:100%; height:auto;">\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Successfully embedded images in eda_report.md")
