from setuptools import setup, find_namespace_packages

setup(
    name="seethrough3d",
    version="0.1.0",
    packages=find_namespace_packages(include=["train*", "inference*"]),
)
