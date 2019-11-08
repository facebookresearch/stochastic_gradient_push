#!/bin/sh

pip install -r requirements.txt

git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 1b903852aecd388e10f03e470fcb1993f1c871dd
python setup.py install

cd ..	
rm -rf apex
