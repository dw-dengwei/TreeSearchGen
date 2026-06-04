import re

def extract_json_from_response(text):
  json_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```')

  # Extracting the JSON block
  json_matches = json_pattern.findall(text)[0]

  json_matches = json_matches.replace("```json", "").replace("```", "")

  return json_matches
