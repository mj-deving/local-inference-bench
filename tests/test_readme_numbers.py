"""Every number in the README's Results and Decision sections must appear in a results file."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
# Structural numbers that describe the setup, not results.
SETUP = {"1", "2", "3", "4", "8", "16", "48", "40", "5", "0", "100"}


def _sections(text: str, names: tuple[str, ...]) -> str:
    out, keep = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line[3:].strip().startswith(names)
        if keep:
            out.append(re.sub(r"^\s*\d+\.\s", "", line))  # ordered-list markers are not results
    return "\n".join(out)


def test_readme_result_numbers_are_backed_by_results_files():
    readme = (ROOT / "README.md").read_text()
    body = _sections(readme, ("Results", "Decision"))
    if not body:
        pytest.skip("README has no Results section yet")
    backing = " ".join((ROOT / "results" / f).read_text() for f in ("summary.md", "summary.json", "cost.json", "mlx-8bit-incident.md"))
    backed = set(NUM.findall(backing))
    missing = sorted({n for n in NUM.findall(body) if n not in SETUP and n not in backed})
    assert not missing, f"README numbers not found in results/: {missing}"
