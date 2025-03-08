import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import mlagents_envs

VERSION = mlagents_envs.__version__
EXPECTED_TAG = mlagents_envs.__release_tag__


# Read the contents of README file for the long description.
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


class VerifyVersionCommand(install):
    """
    Custom command to verify that the git tag is the expected one for the release.
    Originally based on https://circleci.com/blog/continuously-deploying-python-packages-to-pypi-with-circleci/
    This differs slightly because our tags and versions are different.
    """

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("GITHUB_REF", "NO GITHUB TAG!").replace("refs/tags/", "")

        if tag != EXPECTED_TAG:
            info = "Git tag: {} does not match the expected tag of this app: {}".format(
                tag, EXPECTED_TAG
            )
            sys.exit(info)


setup(
    name="safe_riverine_envs",
    version="0.1.0",
    description="A Unity-based vision-driven river following safe reinforcement learning environment, built upon mlagents_envs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdisonPricehan/ml-agents-river",
    author="Zihan Wang",
    author_email="wang5044@purdue.edu",
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        # "Programming Language :: Python :: 3.8",
        # "Programming Language :: Python :: 3.9",
        # "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "colabs", "*.ipynb", ".idea", "images"]
    ),
    zip_safe=False,
    install_requires=[
        "cloudpickle",
        "grpcio>=1.11.0",
        "Pillow>=4.2.1",
        "protobuf>=3.6,<3.20",
        "pyyaml>=3.1.0",
        "gymnasium>=0.29.0",
        "pettingzoo==1.15.0",
        "numpy>=1.21.2",
        "filelock>=3.4.0",
    ],
    python_requires=">=3.8.13,<=3.10.12",
    # TODO: Remove this once mypy stops having spurious setuptools issues.
    # cmdclass={"verify": VerifyVersionCommand},  # type: ignore
)
