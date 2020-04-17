import pandas as pd
import numpy as np

def intake_data(method = 0, upsample = True):
    print("TEST")
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

    # these are the anomalies detected in the exploration notebook file 
    indexes_to_drop = [70, 135, 162, 189]
    for ind in indexes_to_drop:
      data = data.drop(data.index[ind])

    data_x = data.drop(["death","days.1","ventilator weaning = 1", "VFD ","days"], axis=1)
    data_y = data["death"]
    column_names = list(data_x.columns)
    
    if not upsample:
        return (data_x, data_y)
    try:
        from imblearn import over_sampling
    except Exception as e:
        print(f"looks like imblearn isn't installed! skipping upsampling: {str(e)}")
        return (data_x, data_y)
        
    oversample = over_sampling.SMOTE()
    X_SMOTE, Y_SMOTE = oversample.fit_resample(data_x, data_y)
    x_frame = pd.DataFrame(X_SMOTE)
    x_frame.columns = column_names
    return (x_frame, Y_SMOTE)
