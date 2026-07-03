import codecs
from typing import Any,List
import json
import jsonlines

def read_jsonl(path: str) -> List[Any]:
    with jsonlines.open(path) as reader:
        data = [obj for obj in reader]
    return data


def write_jsonl(data: List[Any], path: str):
    with jsonlines.open(path, mode="w") as writer:
        writer.write_all(data)


def read_json(path: str) -> Any:
    with codecs.open(path, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    return data


def write_json(data: Any, path: str, **kwargs):
    with codecs.open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, **kwargs)


def read_txt(path: str) -> Any:
    with codecs.open(path, "r", encoding="utf-8") as fin:
        data = fin.read().splitlines()
    return data


def write_txt(data: List[str], path: str, **kwargs):
    with codecs.open(path, "w", encoding="utf-8") as fout:
        for d in data:
            fout.write(d)
            fout.write("\n")
