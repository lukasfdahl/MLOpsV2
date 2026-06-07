#!/bin/bash
# Run the unit tests with coverage. Produces:
#   - a terminal summary (with missing lines)
#   - coverage_out/coverage.xml  (Cobertura — for CI tools)
#   - coverage_out/htmlcov/      (browsable HTML report)
set -e

mkdir -p coverage_out

echo "Starting Pytest with coverage"
pytest tests/ \
    --cov=src \
    --cov-branch \
    --cov-report=term-missing \
    --cov-report=xml:coverage_out/coverage.xml \
    --cov-report=html:coverage_out/htmlcov
