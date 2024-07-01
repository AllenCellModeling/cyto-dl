# this file is intended to be called by a github workflow (.github/workflows/publish_to_pypi.yaml)
# it makes decisions based on the current version and the component specified for bumping,
# which the workflow cannot do
import subprocess
import sys
from typing import Set, List
import toml


def main():
    if len(sys.argv) < 2:
        raise ValueError("No component specified for bumping version")

    component: str = sys.argv[1].lower()
    valid_options: Set[str] = {"major", "minor", "patch", "dev"}

    if component not in valid_options:
        raise ValueError(f"Component must be one of {valid_options}")

    version: str = toml.load("pyproject.toml")["project"]["version"]
    version_components: List[str] = version.split(".")

    update_output: subprocess.CompletedProcess = None
    # 4 components means we currently have a dev version
    if len(version_components) == 4:
        if component == "dev":
            # increment the dev tag (e.g. 1.0.0.dev0 -> 1.0.0.dev1)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag-num", "-n"]
            )
        elif component == "patch":
            # finalize the patch by removing dev tag (e.g. 1.0.0.dev1 -> 1.0.0)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag=final", "-n"]
            )
        else:
            raise ValueError(
                "Cannot update major or minor version while dev version is current"
            )

    elif len(version_components) == 3:
        if component == "dev":
            # increment patch and begin at dev0 (e.g. 1.0.0 -> 1.0.1.dev0)
            update_output = subprocess.run(
                ["bumpver", "update", "--patch", "--tag=dev", "-n"]
            )
        else:
            update_output = subprocess.run(
                ["bumpver", "update", f"--{component}", "-n"]
            )

    else:
        raise ValueError(
            f"Unknown version format: {version}. Expected MAJOR.MINOR.PATCH[.PYTAGNUM]"
        )

    if update_output.returncode != 0:
        raise RuntimeError(
            f"bumpver exited with code {update_output.returncode}"
        )


if __name__ == "__main__":
    main()

"""
TESTING:
- add and commit any changes (keep track of this commit hash)
- bumpver update --set-version 1.0.0

- python publish_bumpver_handler.py
  - expect: ValueError

- python publish_bumpver_handler.py fake
  - expect: ValueError

- python publish_bumpver_handler.py major
  - expect: version updated to 2.0.0

- python publish_bumpver_handler.py minor
  - expect: version updated to 2.1.0

- python publish_bumpver_handler.py patch
  - expect: version updated to 2.1.1

- python publish_bumpver_handler.py dev
  - expect: version updated to 2.1.2.dev0

- python publish_bumpver_handler.py dev
  - expect: version updated to 2.1.2.dev1

- python publish_bumpver_handler.py major
  - expect: ValueError

- python publish_bumpver_handler.py minor
  - expect: ValueError

- python publish_bumpver_handler.py patch
  - expect: version updated to 2.1.2

- git reset --hard {hash of the commit made at the beginning}
- git tag --delete 1.0.0 2.0.0 2.1.0 2.1.1 2.1.2 2.1.2.dev0 2.1.2.dev1
"""