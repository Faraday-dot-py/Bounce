import hashlib
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"
JS = sorted((WEB / "js").glob("*.js"))
IMPORT = re.compile(r'(from "\./[a-z0-9_]+\.js)(\?v=[0-9a-f]+)?"')
SRC = re.compile(r'(src="js/app\.js)(\?v=[0-9a-f]+)?"')
VQ = re.compile(r'const VQ = "(\?v=[0-9a-f]*)?";')


def strip(text):
    return VQ.sub('const VQ = "";', IMPORT.sub(r'\1"', SRC.sub(r'\1"', text)))


def main():
    files = JS + [WEB / "index.html", WEB / "weights.json", WEB / "weights.bin"]
    h = hashlib.sha1()
    for f in files:
        data = f.read_bytes()
        h.update(strip(data.decode()).encode() if f.suffix in (".js", ".html") else data)
    v = h.hexdigest()[:10]
    for f in JS:
        t = f.read_text()
        t = IMPORT.sub(lambda m: f'{m.group(1)}?v={v}"', t)
        t = VQ.sub(f'const VQ = "?v={v}";', t)
        f.write_text(t)
    idx = WEB / "index.html"
    idx.write_text(SRC.sub(lambda m: f'{m.group(1)}?v={v}"', idx.read_text()))
    print(v)


main()
