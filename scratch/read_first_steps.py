import json

log_path = r"C:\Users\chebo\.gemini\antigravity\brain\5157bc80-9046-4ab8-92d0-532f92d5a1e3\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            source = data.get('source', '')
            if source == 'MODEL' and idx < 100:
                print(f"Step {idx} MODEL:")
                print(data.get('content', ''))
                print("="*60)
            elif source == 'USER_EXPLICIT' and idx < 100:
                print(f"Step {idx} USER:")
                print(data.get('content', ''))
                print("="*60)
        except Exception as e:
            pass
