#!/usr/bin/env python
import os
import sys
import importlib

def run_preflight_checks():
    print("=" * 60)
    print("SMART RESEARCH ASSISTANT - DEPLOYMENT PRE-FLIGHT CHECK")
    print("=" * 60)

    # 1. Check Python Version
    print("[*] Checking Python Version...")
    py_version = sys.version_info
    print(f"    Current Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 11):
        print("    [!] WARNING: Recommended Python version is 3.11+. Some libraries might have issues on older versions.")
    else:
        print("    [+] Python version check passed.")

    # 2. Check Folder Structure
    print("\n[*] Checking Directories...")
    required_dirs = [
        "src",
        "src/ingestion",
        "src/embeddings",
        "src/vectorstores",
        "src/retrieval",
        "src/llm",
        "src/evaluation",
        "src/ui",
        "src/utils",
        "assets",
        "docs",
        "tests",
        "sample_data"
    ]
    all_dirs_exist = True
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"    [+] Directory exists: {d}")
        else:
            print(f"    [-] MISSING Directory: {d}")
            all_dirs_exist = False
    
    if all_dirs_exist:
        print("    [+] Directory structure check passed.")
    else:
        print("    [!] WARNING: Some folders are missing. Please ensure your folder structure matches the specs.")

    # 3. Check configuration files
    print("\n[*] Checking Configuration Files...")
    required_files = [
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "assets/styles.css",
        "assets/logo.png"
    ]
    for f in required_files:
        if os.path.isfile(f):
            print(f"    [+] Config file exists: {f}")
        else:
            print(f"    [-] MISSING Config file: {f}")

    # 4. Check critical imports
    print("\n[*] Checking Critical Dependencies...")
    libs = [
        ("streamlit", "streamlit"),
        ("pydantic", "pydantic"),
        ("dotenv", "python-dotenv"),
        ("pypdf", "pypdf"),
        ("bs4", "beautifulsoup4"),
        ("langchain", "langchain"),
        ("langchain_core", "langchain-core"),
        ("langchain_community", "langchain-community")
    ]
    
    for lib_import, lib_name in libs:
        try:
            importlib.import_module(lib_import)
            print(f"    [+] Dependency '{lib_name}' is installed.")
        except ImportError:
            print(f"    [-] Dependency '{lib_name}' is NOT installed. Install via: pip install -r requirements.txt")

    print("\n" + "=" * 60)
    print("Pre-flight check completed.")
    print("=" * 60)

if __name__ == "__main__":
    run_preflight_checks()
