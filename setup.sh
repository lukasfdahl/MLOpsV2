#!/bin/bash

echo "Setting up MLOps V2 local environment:"
python -m pip install --upgrade pip # Update pip to avoid errors
echo "(Finsihed) Setting up MLOps V2 local environment"

echo "Installing dependencies:"
pip install -r requirements.txt # install developer requirements
echo "(Finsihed) Installing dependencies"

echo "Setting up Git hooks:"
pre-commit install # to setup precommit
echo "(Finsihed) Setting up Git hooks"
