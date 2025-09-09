"""Useful functions for git operations.

(e.g. getting git hash, last commit message, etc.)
"""

import subprocess  # nosec


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")  # nosec


def get_git_status():
    cmd = "git diff -- . ':!*.ipynb' --color"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
    stdout, stderr = process.communicate()
    git_diff_output = stdout.decode("utf-8")
    separator_start = f"\n{100 * '='}\n{'=' * 10} start git diff {'=' * 10}\n"
    separator_end = f"\n{'=' * 10} end git diff {'=' * 10}\n{100 * '='}\n"
    return separator_start + git_diff_output + separator_end


def get_last_commit_message():
    return (
        subprocess.check_output(  # nosec
            ["git", "log", "-1", "--pretty=%B"],
        )
        .strip()
        .decode("utf-8")
    )
