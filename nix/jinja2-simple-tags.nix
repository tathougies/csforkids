{ buildPythonPackage,
  lib,
  fetchPypi,
  jinja2,
  setuptools
}:

buildPythonPackage rec {
  pname = "jinja2-simple-tags";
  version = "0.6.1";

  pyproject = true;
  build-system = [ setuptools ];

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-VKv4OIPc0T+P0uosQv7uqEGN82QJB71SUd7F4lpq8OM=";
  };

  propagatedBuildInputs = [ jinja2 ];

  meta = {
    homepage = "https://github.com/dldevinc/jinja2-simple-tags";
    description = "Base classes for quick-and-easy template tag development";
    license = lib.licenses.bsd3;
  };
}
