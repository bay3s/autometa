from setuptools import setup
import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="autometa",
    install_requires=required,
    packages=setuptools.find_packages(
        exclude=[
            "notes",
        ]
    ),
)
