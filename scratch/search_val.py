import json

log_path = r"C:\Users\chebo\.gemini\antigravity\brain\5157bc80-9046-4ab8-92d0-532f92d5a1e3\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            content = data.get('content', '')
            if 'media_' in content or 'media__' in content or '.png' in content:
                print(f"Step {idx} ({data.get('source')}):")
                print(content[:500])
                print("-" * 50)
        except Exception as e:
            pass
