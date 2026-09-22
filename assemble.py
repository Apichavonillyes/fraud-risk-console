"""Build the dashboard: inject data.json + app.js into template.html.

Writes two files:
  dist/fraud-console.html  - page body only (for embedding)
  docs/index.html          - full standalone page (open locally or serve with GitHub Pages)
"""
import pathlib
root = pathlib.Path(__file__).parent
body = ((root / "template.html").read_text()
        .replace("__DATA__", (root / "data.json").read_text())
        .replace("/*__APP__*/", (root / "app.js").read_text()))
for d in ("dist", "docs"):
    (root / d).mkdir(exist_ok=True)
(root / "dist" / "fraud-console.html").write_text(body)
(root / "docs" / "index.html").write_text(
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"></head>\n<body>\n'
    + body + "\n</body></html>\n")
print("built dist/fraud-console.html and docs/index.html")
