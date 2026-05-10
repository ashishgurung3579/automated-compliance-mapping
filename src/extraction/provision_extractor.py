"""
Parse structured provisions from ETSI EN 303 645 and EN 304 223 text.

EN 303 645 provisions: "Provision 5.X-Y[A-Z]?"
EN 304 223 provisions: "Provision 5.X.Y-Z[.W]?"
"""
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src.extraction.utils.pdf_parser import load_standard


# Section metadata for EN 303 645.
SECTION_TITLES_303645 = {
    "5.0": "Reporting implementation",
    "5.1": "No universal default passwords",
    "5.2": "Implement a means to manage reports of vulnerabilities",
    "5.3": "Keep software updated",
    "5.4": "Securely store sensitive security parameters",
    "5.5": "Communicate securely",
    "5.6": "Minimize exposed attack surfaces",
    "5.7": "Ensure software integrity",
    "5.8": "Ensure that personal data is secure",
    "5.9": "Make systems resilient to outages",
    "5.10": "Examine system telemetry data",
    "5.11": "Make it easy for users to delete user data",
    "5.12": "Make installation and maintenance of devices easy",
    "5.13": "Validate input data",
}

# Principle metadata for EN 304 223.
PRINCIPLE_META_304223 = {
    "5.1.1": ("Raise awareness of AI security threats and risks", "Secure Design"),
    "5.1.2": ("Design the AI system for security as well as functionality and performance", "Secure Design"),
    "5.1.3": ("Evaluate the threats and manage the risks to the AI system", "Secure Design"),
    "5.1.4": ("Enable human responsibility for AI systems", "Secure Design"),
    "5.2.1": ("Identify, track and protect the assets", "Secure Development"),
    "5.2.2": ("Secure the infrastructure", "Secure Development"),
    "5.2.3": ("Secure the supply chain", "Secure Development"),
    "5.2.4": ("Document data, models and prompts", "Secure Development"),
    "5.2.5": ("Conduct appropriate testing and evaluation", "Secure Development"),
    "5.3.1": ("Communication and processes associated with End-users and Affected Entities", "Secure Deployment"),
    "5.4.1": ("Maintain regular security updates, patches and mitigations", "Secure Maintenance"),
    "5.4.2": ("Monitor the system's behaviour", "Secure Maintenance"),
    "5.5.1": ("Ensure proper data and model disposal", "Secure End of Life"),
}


@dataclass
class Provision:
    provision_id: str      # e.g. "5.1-1" or "5.1.1-1"
    standard: str          # "EN 303 645" or "EN 304 223"
    section_id: str        # e.g. "5.1" or "5.1.1"
    title: str
    text: str
    modality: str          # "shall" | "should" | "may" | "unknown"


# EN 303 645 parser.

# Matches: "Provision 5.3-4A" or "Provision 5.10-1"
_PAT_303645 = re.compile(
    r"Provision\s+(5\.\d{1,2}-\d+[A-Z]?)\s+(.*?)(?=Provision\s+5\.\d|^\d+\s+|\Z)",
    re.DOTALL | re.MULTILINE,
)

_VOID_PAT = re.compile(r"^\s*[Vv]oid\.?\s*$", re.MULTILINE)

def _detect_modality(text: str) -> str:
    low = text.lower()
    if "shall not" in low or "shall" in low:
        return "shall"
    if "should not" in low or "should" in low:
        return "should"
    if "may" in low:
        return "may"
    return "unknown"


def _section_from_id(provision_id: str) -> str:
    return re.match(r"(5\.\d+)", provision_id).group(1)


def parse_303645(text: str) -> list[Provision]:
    # Skip the Annex B body; the real annex appears after the table of contents.
    annex_match = re.search(r"\nAnnex B", text[50000:])
    if annex_match:
        text = text[: 50000 + annex_match.start()]

    provisions = []
    for m in _PAT_303645.finditer(text):
        pid = m.group(1)
        body = m.group(2).strip()

        # Drop explanatory blocks so the provision text stays focused.
        body = re.sub(r"\n(NOTE|EXAMPLE)\s+\d*:.*", "", body, flags=re.DOTALL).strip()

        if bool(_VOID_PAT.search(body)) or "Void" in body[:20]:
            continue
        sec = _section_from_id(pid)

        provisions.append(Provision(
            provision_id=pid,
            standard="EN 303 645",
            section_id=sec,
            title=SECTION_TITLES_303645.get(sec, "Unknown"),
            text=body,
            modality=_detect_modality(body),
        ))
    return provisions


# EN 304 223 parser.

# Matches: "Provision 5.1.1-1" or "Provision 5.1.2-1.1"
_PAT_304223 = re.compile(
    r"Provision\s+(5\.\d+\.\d+-\d+(?:\.\d+)?)\s+(.*?)(?=Provision\s+5\.\d|\Z)",
    re.DOTALL | re.MULTILINE,
)

def _principle_key_from_id(provision_id: str) -> str:
    return re.match(r"(5\.\d+\.\d+)", provision_id).group(1)


def parse_304223(text: str) -> list[Provision]:
    provisions = []
    for m in _PAT_304223.finditer(text):
        pid = m.group(1)
        body = m.group(2).strip()

        key = _principle_key_from_id(pid)
        title, _phase = PRINCIPLE_META_304223.get(key, ("Unknown", "Unknown"))

        provisions.append(Provision(
            provision_id=pid,
            standard="EN 304 223",
            section_id=key,
            title=title,
            text=body,
            modality=_detect_modality(body),
        ))
    return provisions


# Save helpers.

def save_provisions(provisions: list, out_path: str | Path) -> None:
    data = [asdict(p) for p in provisions]
    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(data)} provisions to {out_path}")


if __name__ == "__main__":
    base = Path(__file__).parents[2]
    raw_dir = base / "data" / "raw"
    out_dir = base / "data" / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    text_303645 = load_standard(raw_dir / "etsi303645v030103p.pdf")
    provs_303645 = parse_303645(text_303645)
    print(f"\nEN 303 645: {len(provs_303645)} provisions")
    save_provisions(provs_303645, out_dir / "en303645_provisions.json")

    text_304223 = load_standard(raw_dir / "etsi304223v020101p.pdf")
    provs_304223 = parse_304223(text_304223)
    print(f"EN 304 223: {len(provs_304223)} total")
    save_provisions(provs_304223, out_dir / "en304223_provisions.json")
