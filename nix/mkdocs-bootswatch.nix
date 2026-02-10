{ buildPythonPackage,
  lib,
  fetchFromGitHub,
  mkdocs,
  setuptools
}:

buildPythonPackage rec {
  pname = "mkdocs-bootswatch";
  version = "1.0";

  pyproject = true;

  src = fetchFromGitHub {
    owner = "mkdocs";
    repo = "mkdocs-bootswatch";
    rev = "374197dbc3b89b830075b7d67c66c9407968855a";
    hash = "sha256-OqwYT7/a/afDRYYu6u2tjEc0A29jtADzzFH26RN1Fqo=";
  };

  build-system = [ setuptools ];

  propagatedBuildInputs = [ mkdocs ];

  meta = {
    homepage = "https://github.com/mkdocs/mkdocs-swan";
    description = "MkDocs Bootswatch themes";
    license = lib.licenses.mit;
  };
}
