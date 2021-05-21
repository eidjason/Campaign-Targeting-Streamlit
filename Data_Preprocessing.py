import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
from sklearn.preprocessing import OneHotEncoder

def data_preprocessing(data):
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])

    data.fillna(data.select_dtypes(include='number').mean().iloc[0], inplace=True)
    data.fillna(data.select_dtypes(include='object').mode().iloc[0], inplace=True)
    data.fillna(data.select_dtypes(include='datetime').mode().iloc[0], inplace=True)

    #  Create customer's account age
    data["Dt_Customer_year_month"] = data["Dt_Customer"].dt.to_period("M")
    data["Account_age"] = ((pd.to_datetime("2014-12").year - data["Dt_Customer_year_month"].dt.year) * 12
                           + (pd.to_datetime("2014-12").month - data["Dt_Customer_year_month"].dt.month))
    data['Customer_age'] = 2014 - data["Year_Birth"]
    data = data.rename(columns={"ID": "Customer_ID"})
    data_new = data.drop(["Year_Birth", "Dt_Customer", "Dt_Customer_year_month", "Z_CostContact", "Z_Revenue"], axis=1)

    # OneHotEncode Education Column
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(data_new[['Education']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_arr, columns=feature_labels)
    data_new = data_new.join(features)

    # OneHotEncode Marital_Status Column
    feature_arr = ohe.fit_transform(data_new[['Marital_Status']]).toarray()
    feature_labels = ohe.categories_
    feature_labels = np.array(feature_labels).ravel()
    features = pd.DataFrame(feature_arr, columns=feature_labels)
    data_new = data_new.join(features)

    data_new = data_new.drop(["Education", "Marital_Status"], axis=1)

    educations = ["Graduation", "PhD", "Master", "2n Cycle", "Basic"]
    marital_statuses = ["Married", "Together", "Single", "Divorced", "Widow", "Alone", "YOLO", "Absurd"]

    for education in educations:
        if education not in data_new.columns.tolist():
            data_new[education] = 0.0

    for status in marital_statuses:
        if status not in data_new.columns.tolist():
            data_new[status] = 0.0

    #  Calculate total amount spent on products
    data_new["total_amount"] = data_new["MntWines"] + data_new["MntFruits"] + data_new["MntMeatProducts"] + data_new[
        'MntFishProducts'] + data_new["MntSweetProducts"] + data_new["MntGoldProds"]

    #  Out of the total amount spent, calculate the percentage of spends by Item
    data_new['MntWines_percentage'] = data_new['MntWines'] / data_new['total_amount']
    data_new['MntFruits_percentage'] = data_new["MntFruits"] / data_new['total_amount']
    data_new["MntMeatProducts_percentage"] = data_new["MntMeatProducts"] / data_new['total_amount']
    data_new["MntFishProducts_percentage"] = data_new["MntFishProducts"] / data_new['total_amount']
    data_new["MntSweetProducts_percentage"] = data_new["MntSweetProducts"] / data_new['total_amount']
    data_new["MntGoldProds_percentage"] = data_new["MntGoldProds"] / data_new['total_amount']

    # Total number of accepted Campaign offers by customers
    data_new["AcceptedCmps"] = data_new[
        ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1)
    data_new = data_new.drop(
        ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "total_amount", "MntWines",
         "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"], axis=1)

    data_new.reset_index(drop=True, inplace=True)

    return data_new