# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys

from setuptools import find_packages, setup


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        sys.exit("Sorry, Python >=3.6 is required for Gossip.")

    with open("README.md", encoding="utf8") as f:
        readme = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()
    setup(
        name="gossip",
        version='0.1',
        description="Gossip-based distributed optimization algorithms implemented in PyTorch.",
        long_description_content_type="text/markdown",
        long_description=readme,
        url="https://github.com/facebookresearch/stochastic_gradient_push",
        license="Attribution-NonCommercial 4.0 International",
        python_requires=">=3.6",
        packages=find_packages(),
        install_requires=reqs.strip().split("\n"),
        extras_require={
            "parse": [
                "pandas",
                "pytz",
                "matplotlib",
            ]
        },
        scripts=["gossip_sgd.py"],
        keywords=["deep learning", "pytorch", "AI"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
