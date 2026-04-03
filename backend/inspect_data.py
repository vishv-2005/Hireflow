import pandas as pd
import joblib

try:
    df_excel = pd.read_excel('../Super_Resume_Dataset_Rows_1_to_1000.xlsx')
    print("Excel skills mean:", df_excel['Skills'].astype(str).apply(lambda x: len(x.split(','))).mean())
    df_json = pd.read_json('./candidates_data.json')
    print("JSON keys:", df_json.keys())
    
    # Just to check the scaler or feature distribution if any.
    model = joblib.load('model.pkl')
    print(model)
except Exception as e:
    print("Error:", e)
