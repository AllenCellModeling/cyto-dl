# This file is intended to be called by a github workflow
import subprocess
import toml


def main():
    version: str = toml.load("pyproject.toml")["project"]["version"]
    tag_output: subprocess.CompletedProcess = subprocess.run(
        ["git", "tag", f"v{version}"]
    )
    if tag_output.returncode != 0:
        raise RuntimeError("failed to tag")


if __name__ == "__main__":
    main()