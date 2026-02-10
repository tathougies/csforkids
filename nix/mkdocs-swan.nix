{ buildPythonPackage,
  lib,
  fetchFromGitHub,
  mkdocs-material,
  setuptools
}:

buildPythonPackage rec {
  pname = "mkdocs-swan";
  version = "0.4.0";

  pyproject = true;

  src = fetchFromGitHub {
    owner = "swan-cern";
    repo = "mkdocs-swan";
    rev = "v0.4.0";
    hash = "sha256-FZX83i2DSwgIQIIQpEc1JrcmqfQnnWhjK/rbLLolLd8=";
  };

  build-system = [ setuptools ];

  propagatedBuildInputs = [ mkdocs-material ];

  meta = {
    homepage = "https://github.com/swan-cern/mkdocs-swan";
    description = "MkDocs SWAN theme";
    license = lib.licenses.agpl3Only;
  };
}
