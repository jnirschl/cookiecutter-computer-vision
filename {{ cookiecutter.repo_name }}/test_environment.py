import sys

REQUIRED_PYTHON = "{{ cookiecutter.python_version }}"
VALID_PYTHON_VERSIONS = ["3.9", "3.10"]


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON not in VALID_PYTHON_VERSIONS:
        raise ValueError(
            f"This project requires Python {VALID_PYTHON_VERSIONS}. Found: Python {sys.version}"
        )

    print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
