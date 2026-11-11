# Copyright 2026 Linn Khant Thuya
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.abc
import sys

OPTIONAL_MODULES = (
    "burmeseai.app",
    "burmeseai.eval",
    "burmeseai.fertility",
    "burmeseai.retrieve",
    "burmeseai.train",
    "burmeseai.zawgyi",
)


class _AbsentOptionalModules(importlib.abc.MetaPathFinder):
    """Report the sibling modules as missing, even if their files exist."""

    def find_spec(self, fullname, path, target=None):
        for name in OPTIONAL_MODULES:
            if fullname == name or fullname.startswith(name + "."):
                raise ModuleNotFoundError(fullname)
        return None


def _drop_loaded_package():
    for key in list(sys.modules):
        if key == "burmeseai" or key.startswith("burmeseai."):
            del sys.modules[key]


def test_package_imports():
    import burmeseai

    assert isinstance(burmeseai.__version__, str)
    assert burmeseai.__version__


def test_import_succeeds_when_optional_modules_are_absent():
    finder = _AbsentOptionalModules()
    sys.meta_path.insert(0, finder)
    _drop_loaded_package()
    try:
        package = importlib.import_module("burmeseai")
        assert package.__version__
        for name in OPTIONAL_MODULES:
            assert name not in sys.modules
    finally:
        sys.meta_path.remove(finder)
        _drop_loaded_package()
        importlib.import_module("burmeseai")
