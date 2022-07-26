import shutil
import sys
from pathlib import Path

import pytest
from cookiecutter import main

CCDS_ROOT = Path(__file__).parents[1].resolve()

args = {
    "project_name": "DrivenData",
    "author_name": "DrivenData",
    "email": "user@mail.com",
    "open_source_license": "BSD-3",
    "python_version": "3.10",
    "n_classes": "10",
    "color_mode": "rgb",
}


def system_check(basename):
    platform = sys.platform
    if "linux" in platform:
        basename = basename.lower()
    return basename


@pytest.fixture(scope="class", params=[{}, args])
def default_baked_project(tmpdir_factory, request):
    temp = tmpdir_factory.mktemp("data-project")
    out_dir = Path(temp).resolve()

    pytest.param = request.param
    main.cookiecutter(
        str(CCDS_ROOT),
        no_input=True,
        extra_context=pytest.param,
        output_dir=out_dir,
        default_config=True,
    )

    pn = pytest.param.get("project_name") or "project_name"

    # project name gets converted to lower case on Linux but not Mac
    pn = system_check(pn)

    proj = out_dir / pn
    request.cls.path = proj
    yield

    # cleanup after
    shutil.rmtree(out_dir)
