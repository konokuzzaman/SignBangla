"""Read the cells of an .xlsx workbook with the standard library (zipfile + XML), so no spreadsheet package is needed.

    sheets = sheet_names(path)              # ["Training Data", ...]
    rows = read_sheet(path, "Training Data")  # list of rows, each a list of cell values (str, int, float or None)

Handles shared strings, inline strings, rich text runs, numbers and booleans. Dates stay as their serial numbers.
"""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}
_REF = re.compile(r"([A-Z]+)(\d+)")


def column_index(letters):
    """A -> 0, Z -> 25, AA -> 26."""
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _text(el):
    """All text of an <si> or <is> element (plain <t> or rich-text runs <r><t>), phonetic runs left out."""
    return "".join(t.text or "" for t in el.iter(f"{{{NS['m']}}}t") if t not in el.findall("m:rPh/m:t", NS))


def _shared_strings(z):
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return [_text(si) for si in root.findall("m:si", NS)]


def _sheet_paths(z):
    """{sheet name: path inside the zip}, in workbook order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    targets = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", NS)}
    out = {}
    for s in wb.findall("m:sheets/m:sheet", NS):
        t = targets[s.get(f"{{{NS['r']}}}id")].lstrip("/")
        out[s.get("name")] = t if t.startswith("xl/") else "xl/" + t
    return out


def sheet_names(path):
    with zipfile.ZipFile(path) as z:
        return list(_sheet_paths(z))


def _value(c, strings):
    kind, v = c.get("t", "n"), c.find("m:v", NS)
    if kind == "inlineStr":
        is_ = c.find("m:is", NS)
        return _text(is_) if is_ is not None else ""
    if v is None or v.text is None:
        return None
    if kind == "s":
        return strings[int(v.text)]
    if kind == "b":
        return v.text == "1"
    if kind in ("str", "e"):
        return v.text
    x = float(v.text)
    return int(x) if x.is_integer() else x


def read_sheet(path, sheet=None):
    """Rows of one sheet (the first when sheet is None), as lists padded to the same length. Empty rows are kept as
    rows of None so row numbers stay the same as in the spreadsheet (row 1 = index 0)."""
    with zipfile.ZipFile(path) as z:
        paths = _sheet_paths(z)
        name = sheet if sheet is not None else next(iter(paths))
        if name not in paths:
            raise KeyError(f"no sheet {name!r} in {path} (sheets: {', '.join(paths)})")
        strings = _shared_strings(z)
        root = ET.fromstring(z.read(paths[name]))
    rows = {}
    for row in root.findall("m:sheetData/m:row", NS):
        r = int(row.get("r")) - 1
        cells = {}
        for i, c in enumerate(row.findall("m:c", NS)):
            ref = c.get("r")
            col = column_index(_REF.match(ref).group(1)) if ref else i
            cells[col] = _value(c, strings)
        rows[r] = cells
    if not rows:
        return []
    width = max((max(c) + 1 for c in rows.values() if c), default=0)
    return [[rows.get(r, {}).get(col) for col in range(width)] for r in range(max(rows) + 1)]
