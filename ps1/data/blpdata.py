"""
Data class to

"""

import pandas as pd


class Data:

    def __init__(self, filename: str):
        self.raw_data = pd.read_csv(filename)
