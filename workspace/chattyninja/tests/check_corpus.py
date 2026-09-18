# Corpus integrity test. Verifies the extracted fixtures before any render test trusts them.
#
# Invoke from the repo root with python3 workspace/chattyninja/tests/check_corpus.py.
#
# Checks, in order:
# - every .meta.json carries exactly the keys row, kwargs, template_sha256, and the sha matches `<suite>/<suite>.jinja`
# - every .json.zst decompresses to its sibling .json byte for byte
# - every frame carries schema chattyninja-chat-render-row-1 with rendered xor expected_error,
#   and generation_spans are codepoint ranges into rendered

import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
SCHEMA = "chattyninja-chat-render-row-1"

def fail(msg):
    """Prints the failure and exits nonzero."""
    print("FAIL:", msg)
    sys.exit(1)

def main():
    """Runs every corpus integrity check over the extracted fixture tree."""
    metas = sorted(CORPUS.glob("*/*.meta.json"))
    if len(metas) != 106:
        fail(f"expected 106 meta files, found {len(metas)}")
    for m in metas:
        d = json.load(open(m))
        if set(d) != {"row", "kwargs", "template_sha256"}:
            fail(f"{m}: unexpected meta keys {sorted(d)}")
        model = m.parts[-2]
        sha = hashlib.sha256((CORPUS / model / f"{model}.jinja").read_bytes()).hexdigest()
        if d["template_sha256"] != sha:
            fail(f"{m}: template_sha256 mismatch against {model}.jinja")
        plain = m.parent / (m.name[: -len(".meta.json")] + ".json")
        zst = plain.with_suffix(".json.zst")
        if not plain.is_file():
            fail(f"{plain}: missing decompressed frame")
        r = subprocess.run(["zstd", "-dc", str(zst)], capture_output=True)
        if r.returncode != 0:
            fail(f"{zst}: zstd decompress failed")
        if r.stdout != plain.read_bytes():
            fail(f"{zst}: decompressed bytes differ from {plain}")
        fr = json.load(open(plain))
        if fr.get("schema") != SCHEMA:
            fail(f"{plain}: schema {fr.get('schema')!r}")
        has_rendered = "rendered" in fr
        has_error = "expected_error" in fr
        if has_rendered == has_error:
            fail(f"{plain}: rendered/expected_error must be exclusive-or")
        if has_rendered:
            text = fr["rendered"]
            for lo, hi in fr.get("generation_spans", []):
                # Spans index codepoints of rendered, verified against HF recorder output.
                if not (0 <= lo <= hi <= len(text)):
                    fail(f"{plain}: span [{lo},{hi}) outside codepoint range {len(text)}")
    rendered = sum(1 for p in CORPUS.glob("*/*.json")
                   if p.name not in {"rows.json", "generated.json"} and "rendered" in json.load(open(p)))
    if rendered != 90:
        fail(f"expected 90 rendered rows, found {rendered}")
    nerr = sum(1 for p in CORPUS.glob("*/err_*.json")
               if "expected_error" in json.load(open(p)))
    if nerr != 16:
        fail(f"expected 16 expected_error rows, found {nerr}")
    print("corpus OK: 106 frames, 90 rendered rows, 16 error rows, sha and zstd verified")

main()
