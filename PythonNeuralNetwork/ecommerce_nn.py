import numpy as np
import pandas as pd
import os 


def getData():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    data = pd.read_csv(dir_path + '\ecommerce_data.csv', header=0)
    print data
