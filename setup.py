from setuptools import find_packages, setup

# Read requirements, but filter out lines starting with '-e'
with open("requirements.txt") as file:
    requirements = [line for line in file.read().splitlines() if not line.startswith('-e')]

# Extract the GitHub dependency
with open("requirements.txt") as file:
    github_dep = [line for line in file.read().splitlines() if line.startswith('-e')][0]

# Remove '-e ' from the beginning of the GitHub dependency
github_dep = github_dep[3:]

setup(
    name="flexmult",
    version="1.0",
    install_requires=requirements,
    dependency_links=[github_dep],
    packages=find_packages(),
)