import json
import ast
import sys

with open('d:/BAV/BTL_AI/pj/AI_Final.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

has_error = False
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        try:
            # specifically use compile to get line numbers etc
            compile(source, f"<cell {i}>", "exec")
        except SyntaxError as e:
            print(f"Cell {i} (0-indexed) has SyntaxError: {e.msg} at line {e.lineno}")
            print(f"Code:\\n{source}")
            has_error = True

if has_error:
    sys.exit(1)
