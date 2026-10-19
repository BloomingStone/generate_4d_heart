"""极简 xlsx 读取（仅用标准库）：打印所有工作表内容。"""
from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_to_idx(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def read_xlsx(path: str | Path) -> dict[str, list[list[str]]]:
    z = zipfile.ZipFile(path)
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))

    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.get("Id"): r.get("Target") for r in rels}
    sheets: dict[str, list[list[str]]] = {}
    for sh in wb.find("m:sheets", NS):
        name = sh.get("name")
        rid = sh.get(f"{{{NS['m'] if False else 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}}}id")
        target = rel_map[rid].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        ws = ET.fromstring(z.read(target))
        rows: list[list[str]] = []
        for row in ws.iter(f"{{{NS['m']}}}row"):
            cells: dict[int, str] = {}
            for c in row.findall("m:c", NS):
                ref = c.get("r")
                t = c.get("t")
                v = c.find("m:v", NS)
                isv = c.find("m:is", NS)
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr" and isv is not None:
                    val = "".join(x.text or "" for x in isv.iter(f"{{{NS['m']}}}t"))
                elif v is not None:
                    val = v.text or ""
                else:
                    val = ""
                cells[_col_to_idx(ref)] = val
            width = max(cells) + 1 if cells else 0
            rows.append([cells.get(i, "") for i in range(width)])
        sheets[name] = rows
    return sheets


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "temp/剂量统计.xlsx"
    for sheet, rows in read_xlsx(p).items():
        print(f"===== sheet: {sheet}  ({len(rows)} rows) =====")
        for i, row in enumerate(rows):
            print(f"{i:3d} | " + " | ".join(f"{c}" for c in row))
