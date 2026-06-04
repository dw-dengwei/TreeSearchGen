import json
import re

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_string = file.read()

    json_string = re.sub(r'//.*?$|/\*.*?\*/', '', json_string, flags=re.DOTALL | re.MULTILINE)
    return json.loads(json_string)

def json_printf(d, **kwargs):
    formatted_json = json.dumps(d, indent=4)
    print(formatted_json, **kwargs)