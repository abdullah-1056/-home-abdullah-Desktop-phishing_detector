"""
PhishGuard AI - Environment Compatibility Checker
Run this before setup to verify your system is compatible.
Usage: python check_env.py
"""
import sys
import os
import platform
import shutil

PASS  = "✅ PASS"
WARN  = "⚠  WARN"
FAIL  = "❌ FAIL"

results = []

def check(label: str, status: str, detail: str = ""):
    icon = {"PASS": "✅", "WARN": "⚠ ", "FAIL": "❌"}[status]
    line = f"  {icon} {label:<30} {detail}"
    print(line)
    results.append(status)

print()
print("=" * 58)
print("  PhishGuard AI — Environment Compatibility Check")
print("=" * 58)
print()

# ── Python Version ─────────────────────────────────────────────
v = sys.version_info
ver_str = f"{v.major}.{v.minor}.{v.micro}"
if v >= (3, 11):
    check("Python version", "PASS", f"{ver_str} (optimal)")
elif v >= (3, 9):
    check("Python version", "PASS", f"{ver_str}")
else:
    check("Python version", "FAIL", f"{ver_str} — requires 3.9+")

# ── OS ────────────────────────────────────────────────────────
os_name = platform.system()
os_ver  = platform.release()
if os_name == "Windows":
    check("Operating System", "PASS", f"Windows {os_ver}")
elif os_name in ("Linux", "Darwin"):
    check("Operating System", "WARN", f"{os_name} — use start.py instead of .bat files")
else:
    check("Operating System", "WARN", f"{os_name} — untested")

# ── Architecture ─────────────────────────────────────────────
arch = platform.machine()
if arch in ("AMD64", "x86_64"):
    check("CPU Architecture", "PASS", f"{arch} (x86_64 — fully supported)")
elif arch.startswith("ARM") or arch == "arm64":
    check("CPU Architecture", "WARN", f"{arch} — may need special PyTorch build")
else:
    check("CPU Architecture", "WARN", f"{arch}")

# ── RAM ──────────────────────────────────────────────────────
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    avail_gb = psutil.virtual_memory().available / (1024**3)
    if avail_gb >= 3.0:
        check("Available RAM",  "PASS", f"{avail_gb:.1f} GB free (need ≥3 GB for models)")
    elif avail_gb >= 1.5:
        check("Available RAM",  "WARN", f"{avail_gb:.1f} GB free — close to minimum")
    else:
        check("Available RAM",  "FAIL", f"{avail_gb:.1f} GB free — insufficient (need ≥3 GB)")
except ImportError:
    check("Available RAM", "WARN", "psutil not installed — cannot check")

# ── Disk Space ────────────────────────────────────────────────
try:
    disk = shutil.disk_usage(".")
    free_gb = disk.free / (1024**3)
    if free_gb >= 3.0:
        check("Free Disk Space", "PASS", f"{free_gb:.1f} GB (need ≥2 GB for models+venv)")
    elif free_gb >= 1.5:
        check("Free Disk Space", "WARN", f"{free_gb:.1f} GB — tight")
    else:
        check("Free Disk Space", "FAIL", f"{free_gb:.1f} GB — insufficient")
except Exception:
    check("Free Disk Space", "WARN", "Cannot determine")

# ── pip ──────────────────────────────────────────────────────
pip_path = shutil.which("pip") or shutil.which("pip3")
if pip_path:
    check("pip", "PASS", pip_path)
else:
    check("pip", "FAIL", "pip not found in PATH")

# ── venv module ───────────────────────────────────────────────
try:
    import venv
    check("venv module", "PASS", "available")
except ImportError:
    check("venv module", "FAIL", "missing — install python3-venv")

# ── Internet (for model download) ─────────────────────────────
try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=5)
    check("HuggingFace access", "PASS", "reachable (needed for first model download)")
except Exception:
    check("HuggingFace access", "WARN", "unreachable — models must be pre-cached")

# ── Summary ───────────────────────────────────────────────────
n_fail = results.count("FAIL")
n_warn = results.count("WARN")
n_pass = results.count("PASS")

print()
print("=" * 58)
print(f"  Results: {n_pass} passed · {n_warn} warnings · {n_fail} failed")
print()
if n_fail == 0:
    print("  ✅ System is compatible. Run setup.bat (Windows)")
    print("     or: python start.py  (cross-platform)")
elif n_fail <= 1 and n_warn <= 2:
    print("  ⚠  System may work with limitations. Review warnings above.")
else:
    print("  ❌ System has compatibility issues. Resolve failures before setup.")
print()
