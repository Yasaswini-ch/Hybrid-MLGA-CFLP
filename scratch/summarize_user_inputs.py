import json

log_path = r"C:\Users\chebo\.gemini\antigravity\brain\5157bc80-9046-4ab8-92d0-532f92d5a1e3\.system_generated\logs\transcript.jsonl"
with open(log_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        try:
            data = json.loads(line)
            source = data.get('source', '')
            step_type = data.get('type', '')
            content = data.get('content', '')
            if source == 'USER_EXPLICIT' and step_type == 'USER_INPUT':
                print(f"Step {idx} USER_INPUT:")
                lines = content.split('\n')
                # print first 5 lines and last 5 lines
                if len(lines) <= 10:
                    print(content)
                else:
                    print('\n'.join(lines[:5]))
                    print("...")
                    print('\n'.join(lines[-5:]))
                print("-" * 50)
        except Exception as e:
            pass
