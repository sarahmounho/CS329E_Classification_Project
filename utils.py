import pandas as pd
import numpy as np

def intake_data(method=0):
    data_unedited = pd.read_csv("data_unedited.csv")
    data = data_unedited[:197]
    data = data.drop(columns = "Unnamed: 0")
    data = data.drop(columns = [f"Unnamed: {a}" for a in range(22,72)])
    data.replace("n.d.", np.nan, inplace = True)
    data["PIP"] = pd.to_numeric(data["PIP"])
    data["TV"] = pd.to_numeric(data["TV"]) 
    data["sex"] = data["sex"].astype(str)
    data = data.replace(['M', 'F'], [0, 1])
    data["sex"] = pd.to_numeric(data["sex"])
    data.rename(columns = {'death = 1 ': 'death'}, inplace=True)
    imputation_methods = [data.mean, data.median]

    data.fillna(imputation_methods[method](), inplace=True) 
    return data
