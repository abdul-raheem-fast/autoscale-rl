#!/usr/bin/env python3
"""Configure the DVC remote (DagsHub) for this repository.

Usage:
    DAGSHUB_USER=you DAGSHUB_TOKEN=<token> python scripts/setup_dvc_remote.py

Collaborators pushing to a teammate's repo also set DAGSHUB_OWNER=<repo owner>;
otherwise the remote URL is built from DAGSHUB_USER.

Credentials are written to `.dvc/config.local`, which is gitignored by DVC —
they never get committed.
"""
from __future__ import annotations

import os
import subprocess
import sys


def run(*args, check=True, capture=False, secret=False) -> subprocess.CompletedProcess:
    shown = list(args[:-1]) + ["****"] if secret else list(args)
    print("+", " ".join(shown))
    return subprocess.run(args, check=check, shell=False,
                          text=True, capture_output=capture)


def main() -> int:
    user = os.environ.get("DAGSHUB_USER", "").strip()
    token = os.environ.get("DAGSHUB_TOKEN", "").strip()
    repo = os.environ.get("DAGSHUB_REPO", "autoscale-rl").strip()
    # Repo owner may differ from the authenticating user (team collaborators).
    owner = os.environ.get("DAGSHUB_OWNER", "").strip() or user
    if not user or not token:
        sys.exit("Set DAGSHUB_USER and DAGSHUB_TOKEN first "
                 "(see .env.example). On Windows PowerShell: "
                 "$env:DAGSHUB_USER='...'; $env:DAGSHUB_TOKEN='...'")

    url = f"https://dagshub.com/{owner}/{repo}.dvc"
    listing = run("dvc", "remote", "list", check=True, capture=True).stdout
    if "origin" in listing.split():
        run("dvc", "remote", "modify", "origin", "url", url, check=False)
    else:
        run("dvc", "remote", "add", "-d", "origin", url)

    # auth (stored locally, NOT committed). DagsHub's DVC endpoint needs HTTP
    # basic auth explicitly, otherwise credentials are never sent (401).
    run("dvc", "remote", "modify", "origin", "auth", "basic", "--local")
    run("dvc", "remote", "modify", "origin", "user", user, "--local")
    run("dvc", "remote", "modify", "--local", "origin", "password", token, secret=True)

    print("\nDVC remote configured:", url)
    print("Next steps:")
    print("  1) dvc repro      # run the full pipeline from scratch")
    print("  2) dvc push       # push data/models to the DagsHub remote")
    print("  3) git add dvc.lock && git commit -m 'chore: lock pipeline'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
