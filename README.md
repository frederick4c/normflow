# M2 Coursework Submission

This repository contains the submission for the M2 normalizing flows coursework.

## Contents

- `coursework.ipynb`: the full coursework notebook
- `pyproject.toml`: Python dependencies
- `data/`: training, validation, and test CSV files
- `figs/`: generated coursework figures
- `checkpoints/flow_full.pt`: final trained model checkpoint
- `logs/training_curves.json`: saved training curves
- `results.json`: required scalar results and writeup

## Running the Notebook

The notebook is intended to be run top-to-bottom on CPU.

1. Install dependencies from `pyproject.toml`.
2. Open `coursework.ipynb`.
3. Run all cells from start to finish.

The notebook generates the required submission artifacts:

- `results.json`
- `figs/Figure1c.pdf`
- `figs/Figure2a.pdf`
- `figs/Figure2c.pdf`
- `figs/Figure3b.pdf`
- `checkpoints/flow_full.pt`
- `logs/training_curves.json`

## Notes

- The notebook sets a fixed random seed for reproducibility.
- The submission is designed for CPU execution and does not require GPU-specific code.

## AI Declaration
In accordance with the Course Handbook, generative AI was used for debugging code, creating the README.md and generating utility code such as saving json files and plotting utils.
