from main import preproseccing
import pickle 


import pandas as pd

with open("model.pkl", 'rb') as f:
     model = pickle.load(f)


def input_data(hours, scores, Extracurricular_Activities, Sleep_Hours, Sample_Papers_Practiced):
    data = {
        'hours': [hours],
        'scores': [scores],
        'Extracurricular_Activities': [Extracurricular_Activities],
        'Sleep_Hours': [Sleep_Hours],
        'Sample_Papers_Practiced': [Sample_Papers_Practiced],
    }
    df=pd.DataFrame(data)
    return df


df=input_data(4,86 ,"No",6 ,9)


prediction = model.predict(preproseccing.transform(df))
print(prediction)

