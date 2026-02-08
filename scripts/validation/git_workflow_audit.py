#!/usr/bin/env python3
"""Audit git workflow compliance against AGENTS.md branch policy."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List

PROGRAM_BRANCH = "feature/team-research-restructure-plan"
SLICE_BRANCH_PATTERN = re.compile(r"^wp/WP-\d{2}(?:[A-Z])?-[a-z0-9][a-z0-9-]*$")


@dataclass
class CheckResult:
    name: str
    status: str
    message: str


@dataclass
class AuditReport:
    timestamp_utc: str
    current_branch: str
    integration_branch: str
    worktree_clean: bool
    accepted_tags: List[str]
    checks: List[CheckResult]

    @property
    def failed(self) -> List[CheckResult]:
        return [check for check in self.checks if check.status == "fail"]


def _git(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["git", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result


def _current_branch() -> str:
    return _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _worktree_clean() -> bool:
    return _git(["status", "--porcelain"]).stdout.strip() == ""


def _local_branch_exists(branch_name: str) -> bool:
    result = _git(["show-ref", "--verify", f"refs/heads/{branch_name}"], check=False)
    return result.returncode == 0


def _remote_branch_exists(branch_name: str) -> bool:
    result = _git(["ls-remote", "--heads", "origin", branch_name], check=False)
    return bool(result.stdout.strip())


def _accepted_tags() -> List[str]:
    out = _git(["tag", "--list", "accepted/*"]).stdout.strip()
    if not out:
        return []
    return sorted([line.strip() for line in out.splitlines() if line.strip()])


def run_audit() -> AuditReport:
    current = _current_branch()
    clean = _worktree_clean()
    accepted = _accepted_tags()

    integration_exists = _local_branch_exists(PROGRAM_BRANCH) or _remote_branch_exists(PROGRAM_BRANCH)

    checks = [
        CheckResult(
            name="current_branch_is_slice_branch",
            status="pass" if current.startswith("wp/") else "fail",
            message=(
                f"Current branch '{current}' is a slice branch."
                if current.startswith("wp/")
                else f"Current branch '{current}' is not a slice branch (expected prefix 'wp/')."
            ),
        ),
        CheckResult(
            name="slice_branch_name_pattern",
            status="pass" if SLICE_BRANCH_PATTERN.match(current) else "fail",
            message=(
                f"Branch name '{current}' matches wp naming policy."
                if SLICE_BRANCH_PATTERN.match(current)
                else (
                    "Branch name does not match expected pattern "
                    "'wp/WP-<NN>[optional-slice-letter]-<slug>' (lowercase slug)."
                )
            ),
        ),
        CheckResult(
            name="not_on_main_or_integration_branch",
            status="pass" if current not in {"main", PROGRAM_BRANCH} else "fail",
            message=(
                f"Current branch '{current}' is isolated from main/integration branches."
                if current not in {"main", PROGRAM_BRANCH}
                else f"Current branch must not be '{current}' for slice work."
            ),
        ),
        CheckResult(
            name="integration_branch_exists",
            status="pass" if integration_exists else "fail",
            message=(
                f"Integration branch '{PROGRAM_BRANCH}' exists (local or remote)."
                if integration_exists
                else f"Integration branch '{PROGRAM_BRANCH}' was not found locally or on origin."
            ),
        ),
        CheckResult(
            name="worktree_clean_before_handoff",
            status="pass" if clean else "fail",
            message=(
                "Working tree is clean."
                if clean
                else "Working tree has uncommitted changes; commit or stash before handoff/integration."
            ),
        ),
        CheckResult(
            name="accepted_checkpoint_tags_present",
            status="pass" if accepted else "warn",
            message=(
                f"Found accepted checkpoint tags: {', '.join(accepted)}"
                if accepted
                else "No accepted checkpoint tags found yet (accepted/wp-...)."
            ),
        ),
    ]

    return AuditReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        current_branch=current,
        integration_branch=PROGRAM_BRANCH,
        worktree_clean=clean,
        accepted_tags=accepted,
        checks=checks,
    )


def _print_human_report(report: AuditReport) -> None:
    print("Git Workflow Audit")
    print("=" * 60)
    print(f"Timestamp (UTC): {report.timestamp_utc}")
    print(f"Current branch:  {report.current_branch}")
    print(f"Integration:     {report.integration_branch}")
    print(f"Worktree clean:  {report.worktree_clean}")
    print(f"Accepted tags:   {report.accepted_tags if report.accepted_tags else 'none'}")
    print("\nChecks:")
    for check in report.checks:
        print(f"- [{check.status.upper()}] {check.name}: {check.message}")


def _to_json(report: AuditReport) -> str:
    return json.dumps(
        {
            "timestamp_utc": report.timestamp_utc,
            "current_branch": report.current_branch,
            "integration_branch": report.integration_branch,
            "worktree_clean": report.worktree_clean,
            "accepted_tags": report.accepted_tags,
            "checks": [asdict(check) for check in report.checks],
        },
        indent=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", help="Optional path to write JSON report")
    args = parser.parse_args()

    report = run_audit()
    _print_human_report(report)

    if args.json_output:
        with open(args.json_output, "w") as f:
            f.write(_to_json(report))
        print(f"\nSaved JSON report to: {args.json_output}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
