from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent
BLP_DATA_LOC = str(_DATA_PATH / 'ps1_ex4.csv')
LOGIT_DATA_LOC = str(_DATA_PATH / 'ps1_ex3.csv')
MICRO_DATA_LOC = str(_DATA_PATH / 'ps1_ex2.csv')
ALL_DATA_PATHS = [BLP_DATA_LOC, LOGIT_DATA_LOC, MICRO_DATA_LOC]