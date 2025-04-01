import json
from pathlib import Path


def open_jsonl(file: Path | str, mode: str = "r", create_if_not_exists: bool = False):
    if isinstance(file, str):
        file = Path(file)
    if not file.exists() and create_if_not_exists:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()  # Creates an empty file
        print(f"Create jsonl file at: {file}")

    with open(file, mode, encoding="utf-8") as fp:
        data = [json.loads(line) for line in fp.readlines()]
        return data


def save_jsonl(
    json_lines: list[dict],
    file: Path | str,
    mode: str = "w",
    ensure_ascii=True,
):
    with open(file, mode, encoding="utf-8") as fp:
        for json_line in json_lines:
            fp.write(json.dumps(json_line, ensure_ascii=ensure_ascii) + "\n")
