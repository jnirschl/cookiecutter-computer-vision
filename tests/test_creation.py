import os
from subprocess import check_output

import pytest
import yaml
from conftest import system_check


def no_curlies(filepath):
    """Utility to make sure no curly braces appear in a file.
    That is, was Jinja able to render everything?
    """
    with open(filepath, "r") as f:
        data = f.read()

    template_strings = ["{{", "}}", "{%", "%}"]

    template_strings_in_file = [s in data for s in template_strings]
    return not any(template_strings_in_file)


@pytest.mark.usefixtures("default_baked_project")
class TestCookieSetup(object):
    def test_project_name(self):
        project = self.path
        if pytest.param.get("project_name"):
            name = system_check("DrivenData")
            assert project.name == name
        else:
            assert project.name == "project_name"

    def test_author(self):
        setup_ = self.path / "setup.py"
        args = ["python", str(setup_), "--author"]
        p = check_output(args).decode("ascii").strip()
        if pytest.param.get("author_name"):
            assert p == "DrivenData"
        else:
            assert p == "Your name (or your organization/company/team)"

    def test_readme(self):
        readme_path = self.path / "README.md"
        assert readme_path.exists()
        assert no_curlies(readme_path)
        if pytest.param.get("project_name"):
            with open(readme_path) as fin:
                assert "# DrivenData" == next(fin).strip()

    def test_setup(self):
        setup_ = self.path / "setup.py"
        args = ["python", str(setup_), "--version"]
        p = check_output(args).decode("ascii").strip()
        assert p == "0.1.0"

    def test_license(self):
        license_path = self.path / "LICENSE"
        assert license_path.exists()
        assert no_curlies(license_path)

    def test_params(self):
        params_path = self.path / "params.yaml"
        assert params_path.exists()
        assert no_curlies(params_path)

        # test values in params
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        assert params["color_mode"] == "rgb"
        assert params["n_classes"] == 10
        assert params["target_size"] == [224, 224, 3]

    def test_license_type(self):
        setup_ = self.path / "setup.py"
        args = ["python", str(setup_), "--license"]
        p = check_output(args).decode("ascii").strip()
        if pytest.param.get("open_source_license"):
            assert p == "BSD-3"
        else:
            assert p == "MIT"

    def test_requirements(self):
        reqs_path = self.path / "requirements.txt"
        assert reqs_path.exists()
        assert no_curlies(reqs_path)

        reqs_path = self.path / "requirements_dev.txt"
        assert reqs_path.exists()
        assert no_curlies(reqs_path)

        # if pytest.param.get("python_interpreter"):
        #     with open(reqs_path) as fin:
        #         lines = list(map(lambda x: x.strip(), fin.readlines()))
        #     assert "pathlib2" in lines

    def test_makefile(self):
        makefile_path = self.path / "Makefile"
        assert makefile_path.exists()
        assert no_curlies(makefile_path)

    def test_folders(self):
        expected_dirs = [
            "data",
            "data/external",
            "data/interim",
            "data/processed",
            "data/raw",
            "docs",
            "logs",
            "models",
            "models/dev",
            "models/final",
            "notebooks",
            "references",
            "reports",
            "reports/figures",
            "reports/project",
            "results",
            "src",
            "src/data",
            "src/data/mapfile",
            "src/features",
            "src/hpc",
            "src/img",
            "src/img/metrics",
            "src/img/morphology",
            "src/img/segmentation",
            "src/models",
            "src/models/train",
            "src/models/networks",
            "src/visualization",
            ".github",
            ".github/workflows",
        ]

        ignored_dirs = [str(self.path)]

        abs_expected_dirs = [str(self.path / d) for d in expected_dirs]
        abs_dirs, _, _ = list(zip(*os.walk(self.path)))
        assert len(set(abs_expected_dirs + ignored_dirs) - set(abs_dirs)) == 0, print(
            f"SetDiff\t{set(abs_expected_dirs + ignored_dirs).difference(set(abs_dirs))}"
        )
