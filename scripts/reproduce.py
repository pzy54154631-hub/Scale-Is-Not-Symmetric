#!/usr/bin/env python3
"""Canonical verification and reproduction entrypoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "README.md", "pyproject.toml", "environment.yml", "requirements.txt",
        "src/mirage/merge.py", "paper/main.tex", "paper/references.bib",
        "repro/paper_claims.csv", "repro/models.yaml", "repro/SHA256SUMS.txt",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")

    claims_path = ROOT / "repro/paper_claims.csv"
    if claims_path.exists():
        with claims_path.open(newline="", encoding="utf-8") as handle:
            claims = list(csv.DictReader(handle))
        if not claims:
            errors.append("claim matrix is empty")
        for row in claims:
            artifact = ROOT / row["required_artifacts"]
            if not artifact.exists():
                errors.append(f"{row['claim_id']}: missing artifact {row['required_artifacts']}")

    sums = ROOT / "repro/SHA256SUMS.txt"
    checked = 0
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"checksum target missing: {relative}")
            elif sha256(path) != expected:
                errors.append(f"checksum mismatch: {relative}")
            checked += 1

    table_inputs = list((ROOT / "paper/tables").glob("*.tex"))
    if len(table_inputs) < 30:
        errors.append(f"expected at least 30 paper tables, found {len(table_inputs)}")

    canonical = [ROOT / "README.md", ROOT / "environment.yml", ROOT / "repro/models.yaml"]
    for path in canonical:
        text = path.read_text(encoding="utf-8")
        if "/Users/" in text or "/root/autodl" in text:
            warnings.append(f"machine-specific example retained in {path.relative_to(ROOT)}")

    print(f"claims: {len(claims) if claims_path.exists() else 0}")
    print(f"checksums: {checked}")
    print(f"paper tables: {len(table_inputs)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print("verification:", "PASS" if not errors else "FAIL")
    return 0 if not errors else 1


def smoke() -> int:
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    failures = []
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    print(f"compiled {len(files)} Python files")
    if failures:
        print("\n".join(failures))
        return 1
    print("smoke test: PASS")
    return 0


def paper() -> int:
    latexmk = shutil.which("latexmk")
    if not latexmk:
        print("latexmk is not installed", file=sys.stderr)
        return 2
    build = ROOT / "paper/build"
    build.mkdir(exist_ok=True)
    command = [latexmk, "-norc", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build}", "main.tex"]
    return subprocess.run(command, cwd=ROOT / "paper").returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "smoke", "paper", "all"])
    args = parser.parse_args()
    if args.command == "verify":
        return verify()
    if args.command == "smoke":
        return smoke()
    if args.command == "paper":
        return paper()
    for action in (verify, smoke, paper):
        code = action()
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
