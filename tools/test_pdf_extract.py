import json
import os
import sys

# ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from utils.pdf_extractor import extract_pdf_metadata

pdf_path = os.path.join('uploads', 's10844-025-01004-9 (1).pdf')
res = extract_pdf_metadata(pdf_path)
print(json.dumps(res, indent=2, ensure_ascii=False))
