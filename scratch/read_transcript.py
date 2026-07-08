import json

log_path = r"C:\Users\chebo\.gemini\antigravity\brain\5157bc80-9046-4ab8-92d0-532f92d5a1e3\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT':
                content = data.get('content', '')
                if '1040444.375' in content:
                    print(content)
        except Exception as e:
            pass
