# Inter-rater reliability calculator

## Prerequisites
- Python3

## Config recommendation
To use the default configuration, you need to place your data as follows:
- 1 file for all labels, name it `labels.txt` and place it under `data/`
- All the files for the first rater should be in the same folder, name it `rater1` and place it under `data/`
- All the files for the second rater should be in the same folder, name it `rater2` and place it under `data/`

## How to run
1. Clone the repository
2. Create a virtual environment
```bash
python3 -m venv venv
```
3. Activate the virtual environment
```bash
source venv/bin/activate
```
4. Install the requirements
```bash
pip install -r requirements.txt
```
5. Config the application in the file `src/processors/irr_processor.py`
5. Run the application
```bash
python -m src.processor.irr_processor
```

## Assumptions
- There are only 2 raters
- All the data points (questions) are unique
- The data points between the 2 raters are identical
