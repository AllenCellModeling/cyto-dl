#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python aics_im2im/train.py trainer.max_epochs=5 logger=csv

python aics_im2im/train.py trainer.max_epochs=10 logger=csv
