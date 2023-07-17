#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python cyto_ml/train.py trainer.max_epochs=5 logger=csv

python cyto_ml/train.py trainer.max_epochs=10 logger=csv
