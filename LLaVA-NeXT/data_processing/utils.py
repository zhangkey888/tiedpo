import json
from pathlib import Path


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
