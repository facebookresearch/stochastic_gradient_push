#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
pip install -r requirements.txt

git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 1b903852aecd388e10f03e470fcb1993f1c871dd
python setup.py install

cd ..	
rm -rf apex
