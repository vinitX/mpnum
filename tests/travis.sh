# Can one call a command with spaces in arguments directly from .travis.yml?
coverage run --source=mpnum setup.py test --addopts "-m '(not verylong) and (not benchmark)'"
