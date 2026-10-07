"""Constroi fixture_fs deterministica (conteudo + mtime fixos) p/ harness local."""
import os

ROOT = "data/staging/v37_templates/fixture_fs"
TS = 1767225600  # 2026-01-01T00:00:00Z

FILES = {
    "api/openapi.yaml": "openapi: 3.0.0\ninfo:\n  title: Store\n  version: 2.1.0\npaths:\n  /items: {}\n",
    "api/README.md": "# Store API\nVersioned handlers under routes/v1.\n",
    "api/routes/users.js": "const db = require('../db');\nmodule.exports.list = (q, r) => r.json(db.users());\n",
    "api/routes/orders.js": "const db = require('../db');\nmodule.exports.create = (q, r) => r.status(201).json({});\n",
    "api/routes/v1/users.js": "// v1 compat shim\nmodule.exports = require('../users');\n",
    "api/routes/v1/orders.js": "// v1 compat shim\nmodule.exports = require('../orders');\n",
    "api/routes/v1/auth.js": "module.exports.check = (t) => t && t.length > 16;\n",
    "api/middleware/auth.js": "module.exports = (q, r, n) => q.headers.token ? n() : r.status(401).end();\n",
    "notes/2024-01.md": "# Jan\nShipped cart fixes. TODO: audit taxes.\n",
    "notes/2024-02.md": "# Feb\nRefunded order 771. TODO: write runbook.\n",
    "notes/todo.txt": "TODO: audit taxes\nTODO: write runbook\nTODO: rotate keys\n",
    "media/sizes.txt": "logo.png 2.1MB\nbanner.svg 1.2KB\n",
    "media/banner.svg": '<svg xmlns="http://www.w3.org/2000/svg"><text>Sale</text></svg>\n',
    "deep/a/b/c/deep.txt": "bottom of the nest\nlevel 4 reached\nnothing else here\n",
}
DIRS = ["yard/cache"]

for d in DIRS:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
for rel, content in FILES.items():
    p = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
with open(os.path.join(ROOT, "media/logo.png"), "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)  # binario real p/ erro de decode
for base, _, files in os.walk(ROOT):
    os.utime(base, (TS, TS))
    for fn in files:
        os.utime(os.path.join(base, fn), (TS, TS))
n = sum(1 for _, _, fs in os.walk(ROOT) for _ in fs)
print("OK fixture_fs:", n, "arquivos, mtime fixo")
