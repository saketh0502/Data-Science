import numpy as np
import pandas as pd
import csv
import xgboost as xgb
# from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from scipy.stats import skew
# from sklearn import cross_validation, metrics
# from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
# from random import randint as random_int


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def getdummyvars(encodedf, df, cname, fill_na):
    encodedf[cname] = df[cname]
    if fill_na is not None:
        encodedf[cname].fillna(fill_na, inplace=True)
    dummies = pd.get_dummies(encodedf[cname], prefix="_"+cname)
    encodedf = encodedf.join(dummies)
    encodedf = encodedf.drop([cname], axis=1)
    return encodedf


def makedummies(df):
    tempdf = pd.DataFrame(index=df.index)
    tempdf = getdummyvars(tempdf, df, "MSSubClass", None)
    tempdf = getdummyvars(tempdf, df, "MSZoning", None)
    tempdf = getdummyvars(tempdf, df, "LotConfig", None)
    tempdf = getdummyvars(tempdf, df, "Neighborhood", None)
    tempdf = getdummyvars(tempdf, df, "Condition1", None)
    tempdf = getdummyvars(tempdf, df, "Condition2", None)
    tempdf = getdummyvars(tempdf, df, "BldgType", None)
    tempdf = getdummyvars(tempdf, df, "HouseStyle", None)
    tempdf = getdummyvars(tempdf, df, "RoofStyle", None)
    tempdf = getdummyvars(tempdf, df, "RoofMatl", None)
    tempdf = getdummyvars(tempdf, df, "Heating", None)
    tempdf = getdummyvars(tempdf, df, "Exterior1st", "VinylSd")
    tempdf = getdummyvars(tempdf, df, "Exterior2nd", "VinylSd")
    tempdf = getdummyvars(tempdf, df, "Foundation", None)
    tempdf = getdummyvars(tempdf, df, "SaleType", "WD")
    tempdf = getdummyvars(tempdf, df, "SaleCondition", "Normal")
    tempdf = getdummyvars(tempdf, df, "LotShape", None)
    tempdf = getdummyvars(tempdf, df, "LandContour", None)
    tempdf = getdummyvars(tempdf, df, "LandSlope", None)
    tempdf = getdummyvars(tempdf, df, "Electrical", "SBrkr")
    tempdf = getdummyvars(tempdf, df, "GarageType", "None")
    tempdf = getdummyvars(tempdf, df, "GarageQual", "None")
    tempdf = getdummyvars(tempdf, df, "GarageCond", "None")
    tempdf = getdummyvars(tempdf, df, "PoolQC", "None")
    tempdf = getdummyvars(tempdf, df, "PavedDrive", None)
    tempdf = getdummyvars(tempdf, df, "MiscFeature", "None")
    tempdf = getdummyvars(tempdf, df, "Fence", "None")
    tempdf = getdummyvars(tempdf, df, "MoSold", None)
    tempdf = getdummyvars(tempdf, df, "GarageFinish", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtExposure", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtFinType1", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtFinType2", "None")
    tempdf = getdummyvars(tempdf, df, "Functional", "None")
    tempdf = getdummyvars(tempdf, df, "ExterQual", "None")
    tempdf = getdummyvars(tempdf, df, "ExterCond", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtQual", "None")
    tempdf = getdummyvars(tempdf, df, "BsmtCond", "None")
    # By including street and alley encodings:
    # XGBoost score on training set:  0.048088620072
    # Lasso score on training set: 0.101160413666
    tempdf = getdummyvars(tempdf, df, "Street", None)
    tempdf = getdummyvars(tempdf, df, "Alley", None)
    # tempdf = getdummyvars(tempdf, df, "GarageYrBlt", None)
    # tempdf = getdummyvars(tempdf, df, "YearBuilt", None)
    idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
    tempdf.loc[idx, "MasVnrType"] = "BrkFace"
    tempdf = getdummyvars(tempdf, df, "MasVnrType", "None")
    return tempdf


def factorize(tdf, df, column, fill_na=None):
    le = LabelEncoder()
    df[column] = tdf[column]
    if fill_na is not None:
        df[column].fillna(fill_na, inplace=True)
    # else:
    #    df[column].fillna("None", inplace=True)
    le.fit(df[column].unique())
    df[column] = le.transform(df[column])
    return df


def process_data(df, neighborhood_map, bldg_type_map, zone_type_map, qual_map):
    processed_df = pd.DataFrame(index=df.index)
    # For LotFrontage nulls, replacing them with median of lot based on neighbourhood instead of calculating square feet
    lotfn = df["LotFrontage"].groupby(df["Neighborhood"])
    for key, group in lotfn:
        idx = (df["Neighborhood"] == key) & (df["LotFrontage"].isnull())
        processed_df.loc[idx, "LotFrontage"] = group.median()
    '''
    for index in range(len(df)):
        if df.ix[index]["MasVnrType"] == 'CBlock':
            processed_df.set_value(index, 'MasVnrScalPrice', df.ix[index]['MasVnrArea'])
        elif df.ix[index]["MasVnrType"] == 'Stone':
            processed_df.set_value(index, 'MasVnrScalPrice', df.ix[index]['MasVnrArea']*1.2)
        elif df.ix[index]["MasVnrType"] == 'BrkCmn':
            processed_df.set_value(index, 'MasVnrScalPrice', df.ix[index]['MasVnrArea']*1.8)
        elif df.ix[index]["MasVnrType"] == 'BrkFace':
            processed_df.set_value(index, 'MasVnrScalPrice', df.ix[index]['MasVnrArea']*2)
        else:
            processed_df.set_value(index, 'MasVnrScalPrice', 0)
    '''

    processed_df["YrSold"] = df["YrSold"]
    # processed_df["Agesincebuilttosold"] = df["YrSold"] - df["YearBuilt"]
    processed_df["Agesincesold"] = 2010 - df["YrSold"]
    processed_df["Age"] = 2010 - df["YearBuilt"]
    processed_df['LotArea'] = df['LotArea']
    processed_df["MoSold"] = df["MoSold"]
    processed_df["Has_Alley"] = df.Alley.replace({'Grvl': 1, 'Pave': 1, 'Street': 1, 'NA': 0})
    processed_df["NeighborhoodIndicator"] = df["Neighborhood"].map(neighborhood_map)
    processed_df["BldgTypeIndicator"] = df["BldgType"].map(bldg_type_map)
    processed_df = factorize(df, processed_df, "HouseStyle")
    processed_df = factorize(df, processed_df, "Condition1")

    processed_df["TotalBsmtSF"] = df["TotalBsmtSF"]
    processed_df["GrLivArea"] = df["GrLivArea"]
    processed_df["Fireplaces"] = df["Fireplaces"]
    processed_df["GarageArea"] = df["GarageArea"]
    processed_df["GarageArea"].fillna(0, inplace=True)

    processed_df["WoodDeckSF"] = df["WoodDeckSF"]
    processed_df["OpenPorchSF"] = df["OpenPorchSF"]
    processed_df["EnclosedPorch"] = df["EnclosedPorch"]
    processed_df["3SsnPorch"] = df["3SsnPorch"]
    processed_df["ScreenPorch"] = df["ScreenPorch"]

    processed_df["PoolArea"] = df["PoolArea"]
    processed_df["MiscVal"] = df["MiscVal"]

    processed_df = factorize(df, processed_df, "MSZoning", "RL")
    processed_df["NewHome"] = df["MSSubClass"].replace({20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
                                                        90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

    processed_df["CentralAir"] = (df["CentralAir"] == "Y") * 1.0

    processed_df["BsmtFullBath"] = df["BsmtFullBath"]
    processed_df["BsmtFullBath"].fillna(0, inplace=True)
    processed_df["BsmtHalfBath"] = df["BsmtHalfBath"]
    processed_df["BsmtHalfBath"].fillna(0, inplace=True)

    processed_df["FullBath"] = df["FullBath"]
    processed_df["HalfBath"] = df["HalfBath"]
    processed_df["BedroomAbvGr"] = df["BedroomAbvGr"]
    processed_df["KitchenAbvGr"] = df["KitchenAbvGr"]
    processed_df["TotRmsAbvGrd"] = df["TotRmsAbvGrd"]

    processed_df["ExterQual"] = df["ExterQual"].map(qual_map).astype(int)
    processed_df["ExterCond"] = df["ExterCond"].map(qual_map).astype(int)
    processed_df["BsmtQual"] = df["BsmtQual"].map(qual_map).astype(int)
    processed_df["BsmtCond"] = df["BsmtCond"].map(qual_map).astype(int)
    processed_df["HeatingQC"] = df["HeatingQC"].map(qual_map).astype(int)
    processed_df["KitchenQual"] = df["KitchenQual"].map(qual_map).astype(int)
    processed_df["FireplaceQu"] = df["FireplaceQu"].map(qual_map).astype(int)
    processed_df["GarageQual"] = df["GarageQual"].map(qual_map).astype(int)
    processed_df["GarageCond"] = df["GarageCond"].map(qual_map).astype(int)
    processed_df["GarageCars"] = df["GarageCars"]
    processed_df["GarageCars"].fillna(0, inplace=True)

    processed_df["BsmtExposure"] = df["BsmtExposure"].map(
        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    processed_df["BsmtFinType1"] = df["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
    processed_df["BsmtFinType2"] = df["BsmtFinType2"].map(bsmt_fin_dict).astype(int)
    # before the lowqual feature: XGBoost score on training set:  0.0483000905975
    #  Lasso score on training set: 0.101562613038

    # XGBoost score on training set:  0.0482947079045
    # Lasso score on training set: 0.101345900977

    # XGBoost score on training set:  0.0482437963479
    # Lasso score on training set: 0.101097066219

    # processed_df["LowQualFinSF"] = df["LowQualFinSF"]

    processed_df["Functional"] = df["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

    processed_df["GarageFinish"] = df["GarageFinish"].map(
        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

    processed_df["Fence"] = df["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
    processed_df["SaleCondition"] = df["SaleCondition"].map({"Abnorml": 1, "AdjLand": 0, "Alloca": 2, "Family": 2,
                                                            "Normal": 2, "Partial": 3}).astype(int)
    processed_df["PavedDrive"] = df["PavedDrive"].map({"Y": 1, "N": 0, "P": 0}).astype(int)

    # New area features:
    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
    processed_df["*TotalArea"] = df[area_cols].sum(axis=1)
    # processed_df["*MainArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    processed_df["*FloorArea"] = df["1stFlrSF"] + df["2ndFlrSF"]
    processed_df["*TotBathrooms"] = df["BsmtFullBath"] + (df["BsmtHalfBath"]) + df["FullBath"] + (
                                       df["HalfBath"])

    processed_df["OverallQual"] = df["OverallQual"]
    processed_df["OverallCond"] = df["OverallCond"]
    # binning data as new features:
    processed_df["*OverallQual"] = df["OverallQual"].map({1: 1, 2: 1, 3: 1,
                                                          4: 2, 5: 2, 6: 2,
                                                          7: 3, 8: 3, 9: 3, 10: 3})
    processed_df["*OverallCond"] = df["OverallCond"].map({1: 1, 2: 1, 3: 1,
                                                          4: 2, 5: 2, 6: 2,
                                                          7: 3, 8: 3, 9: 3, 10: 3})
    # Type of season as a feature
    processed_df["SeasonSold"] = df["MoSold"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                                   6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).astype(int)
    # Time since last remodel as a feature
    processed_df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # adding a high season feature based on simple hist plot of houses sold
    processed_df["HighlySeasonal"] = df["MoSold"].replace(
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    # adding a feature for checking if house was sold before it was completed
    processed_df["WasHouseNotCompleted"] = df["SaleCondition"].replace({"Abnorml": 0, "Alloca": 0,
                                                                        "AdjLand": 0, "Family": 0,
                                                                        "Normal": 0, "Partial": 1})

    processed_df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    processed_df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    processed_df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    processed_df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    processed_df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    processed_df["Neighborhood_Good"].fillna(0, inplace=True)

    processed_df["SaleCondition_PD"] = df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})
    processed_df["BadHeating"] = df.HeatingQC.replace(
        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
    processed_df["GarageYrBlt"] = df["GarageYrBlt"]
    processed_df["GarageYrBlt"].fillna(0.0, inplace=True)
    '''
    # Taking polynomials of top 10 features got from corr.py as new features:
    # x2:
    processed_df["OverallQual2"] = processed_df["OverallQual"]**2
    processed_df["*TotalArea2"] = processed_df["*TotalArea"]**2
    processed_df["NeighborhoodIndicator2"] = processed_df["NeighborhoodIndicator"]**2
    processed_df["*FloorArea2"] = processed_df["*FloorArea"]**2
    processed_df["GrLivArea2"] = processed_df["GrLivArea"]**2
    processed_df["*OverallQual2"] = processed_df["*OverallQual"]**2
    processed_df["ExterQual2"] = processed_df["ExterQual"]**2
    processed_df["*TotBathrooms2"] = processed_df["*TotBathrooms"]**2
    processed_df["KitchenQual2"] = processed_df["KitchenQual"]**2
    processed_df["GarageArea2"] = processed_df["GarageArea"]**2
    # x3:
    processed_df["OverallQual3"] = processed_df["OverallQual"] ** 3
    processed_df["*TotalArea3"] = processed_df["*TotalArea"] ** 3
    processed_df["NeighborhoodIndicator3"] = processed_df["NeighborhoodIndicator"] ** 3
    processed_df["*FloorArea3"] = processed_df["*FloorArea"] ** 3
    processed_df["GrLivArea3"] = processed_df["GrLivArea"] ** 3
    processed_df["*OverallQual3"] = processed_df["*OverallQual"] ** 3
    processed_df["ExterQual3"] = processed_df["ExterQual"] ** 3
    processed_df["*TotBathrooms3"] = processed_df["*TotBathrooms"] ** 3
    processed_df["KitchenQual3"] = processed_df["KitchenQual"] ** 3
    processed_df["GarageArea3"] = processed_df["GarageArea"] ** 3
    # sqrt(x):
    processed_df["root_OverallQual"] = np.sqrt(processed_df["OverallQual"])
    processed_df["root_*TotalArea"] = np.sqrt(processed_df["*TotalArea"])
    processed_df["root_NeighborhoodIndicator"] = np.sqrt(processed_df["NeighborhoodIndicator"])
    processed_df["root_*FloorArea"] = np.sqrt(processed_df["*FloorArea"])
    processed_df["root_GrLivArea"] = np.sqrt(processed_df["GrLivArea"])
    processed_df["root_*OverallQual2"] = np.sqrt(processed_df["*OverallQual"])
    processed_df["root_ExterQual"] = np.sqrt(processed_df["ExterQual"])
    processed_df["root_*TotBathrooms"] = np.sqrt(processed_df["*TotBathrooms"])
    processed_df["root_KitchenQual"] = np.sqrt(processed_df["KitchenQual"])
    processed_df["root_GarageArea"] = np.sqrt(processed_df["GarageArea"])
    '''

    # new2
    processed_df = factorize(df, processed_df, "LotConfig")
    processed_df = factorize(df, processed_df,  "Neighborhood")
    processed_df = factorize(df, processed_df, "RoofStyle")
    processed_df = factorize(df, processed_df, "Exterior1st", "Oth1")
    processed_df = factorize(df, processed_df, "Exterior2nd", "Oth1")
    processed_df = factorize(df, processed_df, "MasVnrType", "None")
    processed_df = factorize(df, processed_df, "Foundation")
    processed_df = factorize(df, processed_df, "SaleType", "Random")

    processed_df["IsRegularLotShape"] = (df["LotShape"] == "Reg") * 1
    processed_df["IsLandLevel"] = (df["LandContour"] == "Lvl") * 1
    processed_df["IsLandSlopeGentle"] = (df["LandSlope"] == "Gtl") * 1
    processed_df["IsElectricalSBrkr"] = (df["Electrical"] == "SBrkr") * 1
    processed_df["IsGarageDetached"] = (df["GarageType"] == "Detchd") * 1
    processed_df["IsPavedDrive"] = (df["PavedDrive"] == "Y") * 1
    processed_df["HasShed"] = (df["MiscFeature"] == "Shed") * 1.
    processed_df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]) * 1
    processed_df["RecentRemodel"] = (df["YearRemodAdd"] == df["YrSold"]) * 1
    processed_df["NewHouse"] = (df["YearBuilt"] == df["YrSold"]) * 1

    processed_df["Has2ndFloor"] = (df["2ndFlrSF"] == 0) * 1
    processed_df["HasMasVnr"] = (df["MasVnrArea"] == 0) * 1
    processed_df["HasWoodDeck"] = (df["WoodDeckSF"] == 0) * 1
    processed_df["HasOpenPorch"] = (df["OpenPorchSF"] == 0) * 1
    processed_df["HasEnclosedPorch"] = (df["EnclosedPorch"] == 0) * 1
    processed_df["Has3SsnPorch"] = (df["3SsnPorch"] == 0) * 1
    processed_df["HasScreenPorch"] = (df["ScreenPorch"] == 0) * 1

    # new 3(filling with binned values, i.e simplifying some of the existing features)
    processed_df["PoolQC"] = df["PoolQC"].map(qual_map).astype(int)
    processed_df["SPoolQC"] = processed_df.PoolQC.replace({1: 1, 2: 1, 3: 2, 4: 2})
    processed_df["SGarageCond"] = processed_df.GarageCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SGarageQual"] = processed_df.GarageQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SFireplaceQu"] = processed_df.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SFireplaceQu"] = processed_df.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SFunctional"] = processed_df.Functional.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
    processed_df["SKitchenQual"] = processed_df.KitchenQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SHeatingQC"] = processed_df.HeatingQC.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SBsmtFinType1"] = processed_df.BsmtFinType1.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
    processed_df["SBsmtFinType2"] = processed_df.BsmtFinType2.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
    processed_df["SBsmtCond"] = processed_df.BsmtCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SBsmtQual"] = processed_df.BsmtQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SExterCond"] = processed_df.ExterCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    processed_df["SExterQual"] = processed_df.ExterQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})

    # Converting any spare nulls to 0s:
    for col in processed_df.columns.values:
        processed_df[col].fillna(0, inplace=True)
    return processed_df

'''
def xgbr(train, trainlabel, test):
    # XGB:
    proc_train_df = train
    train_price_df = trainlabel
    proc_test_df = test
    bag_of_preds = []
    for model in range(20):
        # creating data for a classifier
        bagged_train_data_index = []
        for index in range(len(proc_train_df)):
            curr_data_index = random_int(0, (len(proc_train_df)-1))  # creating a random number
            bagged_train_data_index.append(curr_data_index)  # adding data at index of random number
        proc_train_df_bag = proc_train_df.ix[bagged_train_data_index]
        train_price_df_bag = train_price_df.ix[bagged_train_data_index]
        # creating and training a decision tree
        regr = xgb.XGBRegressor(
            colsample_bytree=0.2,
            gamma=0.0,
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=1.5,
            n_estimators=10000,
            reg_alpha=0.5,
            reg_lambda=0.6,
            subsample=0.2,
            seed=42,
            silent=1)
        regr.fit(proc_train_df_bag, train_price_df_bag)
        bag_of_preds.append(regr)
    y_pred = None
    y_pred_t = None
    for reg in bag_of_preds:
        if y_pred is None:
            y_pred = np.array(reg.predict(proc_train_df))
            y_pred_t = np.array(reg.predict(proc_test_df))
        else:
            y_pred = y_pred + np.array(reg.predict(proc_train_df))
            y_pred_t = y_pred_t + np.array(reg.predict(proc_test_df))
    y_pred = y_pred/len(bag_of_preds)
    y_pred_t = y_pred_t/len(bag_of_preds)
    y_pred = regr.predict(proc_train_df)
    y_test = train_price_df
    y_pred_t = regr.predict(proc_test_df)
    # y_pred_t = np.exp(y_pred_t)
    print("XGBoost score on training set: ", rmse(y_test, y_pred))
    return y_pred_t
'''


def xgbr(train, trainlabel, test):
    # XGB:
    proc_train_df = train
    train_price_df = trainlabel
    proc_test_df = test
    regr = xgb.XGBRegressor(
        colsample_bytree=0.2,
        gamma=0.0,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1.5,
        n_estimators=7200,
        reg_alpha=0.7,
        reg_lambda=0.6,
        subsample=0.2,
        seed=42,
        silent=1)
    regr.fit(proc_train_df, train_price_df)
    y_test = regr.predict(proc_train_df)
    y_pred = train_price_df
    y_pred_t = regr.predict(proc_test_df)
    # y_pred_t = np.exp(y_pred_t)
    print("XGBoost score on training set: ", rmse(y_pred, y_test))
    return y_pred_t


'''
# XGB with CV:
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_params()
        xgtrain = xgb.DMatrix(dtrain, label=predictors)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fitting the algorithm to the data:
    alg.fit(dtrain, predictors, eval_metric='rmse')
    # Predicting training set:
    dtrain_predictions = alg.predict(dtrain)
    # dtrain_predprob = alg.predict_proba(dtrain)[:, 1]
    # Printing model report:
    # print("Accuracy : %.4g" % metrics.accuracy_score(predictors, dtrain_predictions))
    print("rmse (Train): %f" % metrics.mean_squared_error(predictors, dtrain_predictions))
xgb1 = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=8000, silent=True,colsample_bytree=0.4)
modelfit(xgb1, proc_train_df, train_price_df)
'''

'''
def lassor(train, trainlabel, test):
    # Lasso:
    best_alpha = 0.00099
    # best_alpha = 0.0007
    proc_train_df = train
    train_price_df = trainlabel
    proc_test_df = test
    l_bag_of_preds = []
    for model in range(10):
        # creating data for a classifier
        bagged_train_data_index = []
        bagged_train_label = []
        for index in range(len(proc_train_df)):
            curr_data_index = random_int(0, (len(proc_train_df)-1))  # creating a random number
            bagged_train_data_index.append(curr_data_index)  # adding data at index of random number
        proc_train_df_bag = proc_train_df.iloc[bagged_train_data_index]
        # print(bagged_train_data_index)
        train_price_df_bag = train_price_df.iloc[bagged_train_data_index]
        # print(train_price_df_bag)
        # creating and training a decision tree
        regr = Lasso(alpha=best_alpha, max_iter=50000)
        regr.fit(proc_train_df_bag, train_price_df_bag)
        l_bag_of_preds.append(regr)
    # Run prediction on training set to get a rough idea of how well it does.
    ly_pred = None
    ly_pred_t = None
    for reg in l_bag_of_preds:
        if ly_pred is None:
            ly_pred = np.array(reg.predict(proc_train_df))
            ly_pred_t = np.array(reg.predict(proc_test_df))
        else:
            ly_pred = ly_pred + np.array(reg.predict(proc_train_df))
            ly_pred_t = ly_pred_t + np.array(reg.predict(proc_test_df))
    ly_pred = ly_pred/len(l_bag_of_preds)
    ly_pred_t = ly_pred_t/len(l_bag_of_preds)
    ly_test = train_price_df
    # print(regr.coef_)
    coefs = None
    for reg in l_bag_of_preds:
        if coefs is None:
            coefs = reg.coef_
        else:
            coefs = coefs + reg.coef_
        # print(coefs)
    print("Lasso score on training set: " + str(rmse(ly_test, ly_pred)))
    coefs = coefs/len(l_bag_of_preds)
    # print(coefs)
    coefs = np.reshape(coefs, len(coefs.transpose()))
    coefs = pd.Series(coefs, index=proc_train_df.columns)
    # print(coefs.describe)
    pltcoefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    print(pltcoefs)
    pltcoefs.plot(kind="barh")
    plt.title("Coefficients in the lasso model:")
    plt.show()
    # ly_pred_t = np.exp(ly_pred_t)
    return ly_pred_t
'''


# Lasso with CV to find the best alpha:
def lassor(train, trainlabel, test):
    proc_train_df = np.array(train)
    train_price_df = np.array(trainlabel)
    # print(train_price_df.shape)
    train_price_df = np.reshape(train_price_df, len(train_price_df))
    # train_price_df = np.reshape(train_price_df, len(train_price_df))
    proc_test_df = np.array(test)

    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06,
                            0.1, 0.3, 0.6, 1], max_iter=50000, cv=10)

    lasso.fit(proc_train_df, train_price_df)
    alpha = lasso.alpha_
    print("Best alpha: ", alpha)
    print("Centering for more precision around alpha: ", alpha)
    lasso = LassoCV(alphas=[alpha * 0.6, alpha * 0.65, alpha * 0.7, alpha * 0.75, alpha * 0.8, alpha * 0.85,
                            alpha * 0.9, alpha * 0.95, alpha, alpha * 1.05, alpha * 1.10, alpha * 1.15,
                            alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], max_iter=50000, cv=10)
    lasso = lasso.fit(proc_train_df, train_price_df)
    alpha = lasso.alpha_
    print("Best alpha: ", alpha)
    ly_test = train_price_df
    ly_pred = lasso.predict(proc_train_df)
    ly_pred_t = lasso.predict(proc_test_df)
    print("Lasso score on training set: " + str(rmse(ly_test, ly_pred)))
    coefs = np.reshape(lasso.coef_, len(lasso.coef_.transpose()))
    coefs = pd.Series(coefs, index=train.columns)
    pltcoefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    print(pltcoefs)
    pltcoefs.plot(kind="barh")
    plt.title("Coefficients for the lasso model:")
    plt.show()
    # ly_pred_t = np.exp(ly_pred_t)
    return ly_pred_t

'''
#  Ridge:
R_bag_of_preds = []
for model in range(10):
    # creating data for a classifer
    bagged_train_data_index = []
    bagged_train_label = []
    for index in range(len(proc_train_df)):
        curr_data_index = random_int(0, (len(proc_train_df)-1))  # creating a random number
        bagged_train_data_index.append(curr_data_index)  # adding data at index of random number
    proc_train_df_bag = proc_train_df.iloc[bagged_train_data_index]
    # print(bagged_train_data_index)
    train_price_df_bag = train_price_df.iloc[bagged_train_data_index]
    # print(train_price_df_bag)
    # creating and training a decision tree
    regr = Ridge(alpha=0.5, max_iter=10000)
    regr.fit(proc_train_df_bag, train_price_df_bag)
    R_bag_of_preds.append(regr)
# Run prediction on training set to get a rough idea of how well it does.
ry_pred = None
ry_pred_t = None
for reg in R_bag_of_preds:
    if ry_pred is None:
        ry_pred = np.array(reg.predict(proc_train_df))
        ry_pred_t = np.array(reg.predict(proc_test_df))
    else:
        ry_pred = ry_pred + np.array(reg.predict(proc_train_df))
        ry_pred_t = ry_pred_t + np.array(reg.predict(proc_test_df))
ry_pred = ry_pred/len(R_bag_of_preds)
ry_pred_t = ry_pred_t/len(R_bag_of_preds)
ry_test = train_price_df
# print(len(ry_test))
print("Ridge score on training set: " + str(rmse(ry_test, ry_pred)))
# ly_test = np.exp(ly_test)
# ly_pred = np.exp(ly_pred)
ry_pred_t = np.exp(ry_pred_t)
'''


# Ridge:
def ridger(train, trainlabel, test):
    proc_train_df = train
    train_price_df = trainlabel
    proc_test_df = test
    rclf = RidgeCV(alphas=[0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    rclf.fit(proc_train_df, train_price_df)
    alpha = rclf.alpha_
    print("Best alpha:", alpha)
    print("Trying for more precision centered around " + str(alpha))
    rclf = RidgeCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85,
                           alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                           alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], cv=10)
    rclf.fit(proc_train_df, train_price_df)
    alpha = rclf.alpha_
    print("Best alpha:", alpha)
    # print(rclf.coef_.shape)
    coefs = np.reshape(rclf.coef_, len(rclf.coef_.transpose()))
    coefs = pd.Series(coefs, index=proc_train_df.columns)
    print("Ridge regularization picked: " + str(sum(coefs != 0)) + " and eliminated:" + str(sum(coefs == 0)) + " features")
    # print(coefs)
    rvalid = np.array(rclf.predict(proc_train_df))
    # rvalid = np.exp(rvalid)
    rtest = np.array(rclf.predict(proc_test_df))
    # rtest = np.exp(rtest)
    print("Ridge score on training set: " + str(rmse(train_price_df, rvalid)))
    pltcoefs = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    print(pltcoefs)
    pltcoefs.plot(kind="barh")
    plt.title("Coefficients in the Ridge model:")
    plt.show()
    # r_pred_t = np.exp(r_pred_t)
    # print(type(r_pred_t))
    # print(type(r_pred_t))
    # print(r_pred_t[0])

    return rtest


if __name__ == "__main__":

    train_df = pd.read_csv("train_orig.csv")
    test_df = pd.read_csv("test.csv")
    # This row has "2207" as built year, replacing it with median when comparing with house built year
    test_df.loc[1132, "GarageYrBlt"] = 2006
    # Imputing data(with median value) for 2 test observations which have some garage features but missing others:
    test_df.loc[666, "GarageQual"] = "TA"
    test_df.loc[666, "GarageCond"] = "TA"
    test_df.loc[666, "GarageFinish"] = "Unf"
    test_df.loc[666, "GarageYrBlt"] = 1980
    test_df.loc[1116, "GarageType"] = np.nan

    neighborhood_map = {
        "MeadowV": 0,  # 88000
        "IDOTRR": 1,  # 103000
        "BrDale": 1,  # 106000
        "OldTown": 1,  # 119000
        "Edwards": 1,  # 119500
        "BrkSide": 1,  # 124300
        "Sawyer": 1,  # 135000
        "Blueste": 1,  # 137500
        "SWISU": 2,  # 139500
        "NAmes": 2,  # 140000
        "NPkVill": 2,  # 146000
        "Mitchel": 2,  # 153500
        "SawyerW": 2,  # 179900
        "Gilbert": 2,  # 181000
        "NWAmes": 2,  # 182900
        "Blmngtn": 2,  # 191000
        "CollgCr": 2,  # 197200
        "ClearCr": 3,  # 200250
        "Crawfor": 3,  # 200624
        "Veenker": 3,  # 218000
        "Somerst": 3,  # 225500
        "Timber": 3,  # 228475
        "StoneBr": 4,  # 278000
        "NoRidge": 4,  # 290000
        "NridgHt": 4,  # 315000
    }

    bldg_type_map = {
        '2fmCon': 1,  # 127500
        'Duplex': 2,  # 135980
        'Twnhs': 2,  # 137500
        '1Fam': 3,  # 167900
        'TwnhsE': 3  # 172200
    }

    zone_type_map = {
        'A': 1,  # Agriculture
        'C': 2,  # Commercial
        'FV': 3,  # Floating Village Residential
        'I': 4,  # Industrial
        'RH': 5,  # Residential High Density
        'RL': 6,  # Residential Low Density
        'RP': 7,  # Residential Low Density Park
        'RM': 8  # Residential Medium Density
    }
    qual_map = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    proc_train_df = process_data(train_df, neighborhood_map, bldg_type_map, zone_type_map, qual_map)
    proc_test_df = process_data(test_df, neighborhood_map, bldg_type_map, zone_type_map, qual_map)

    train_price_df = pd.DataFrame(index=train_df.index, columns=["SalePrice"])
    train_price_df['SalePrice'] = np.log(train_df['SalePrice'])

    # Dropping few outliers from training set
    proc_train_df = proc_train_df.drop([523, 691, 1182, 1298])
    train_price_df = train_price_df.drop([523, 691, 1182, 1298])
    numfeats = proc_train_df.dtypes[proc_train_df.dtypes != "object"].index

    # Check on it later
    # skew
    skewed = proc_train_df[numfeats].apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > 0.75]
    skewed = skewed.index
    proc_train_df[skewed] = np.log1p(proc_train_df[skewed])
    proc_test_df[skewed] = np.log1p(proc_test_df[skewed])

    # Scaling the numerical features:
    scaler = StandardScaler()
    scaler.fit(proc_train_df[numfeats])
    scaled = scaler.transform(proc_train_df[numfeats])
    for i, col in enumerate(numfeats):
        proc_train_df[col] = scaled[:, i]
    # print(proc_train_df.head(5))
    scaled = scaler.transform(proc_test_df[numfeats])
    for i, col in enumerate(numfeats):
        proc_test_df[col] = scaled[:, i]
    # print(proc_test_df.head(5))
    # print(sum(proc_train_df["LotFrontage"])/len(proc_train_df["LotFrontage"]))
    # Dropping few outliers
    train_df = train_df.drop([523, 691, 1182, 1298])

    # Creating one hot encodings of the categorical columns:
    cattraindf = makedummies(train_df)
    cattestdf = makedummies(test_df)
    proc_train_df = proc_train_df.join(cattraindf)
    proc_test_df = proc_test_df.join(cattestdf)

    print(len(proc_train_df))
    print(len(proc_test_df))

    # Following columns are missing in test data so dropping them
    drop_cols = [
        "_Exterior1st_ImStucc", "_Exterior1st_Stone",
        "_Exterior2nd_Other", "_HouseStyle_2.5Fin",
        "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",
        "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",
        "_Heating_Floor", "_Heating_OthW",
        "_Electrical_Mix",
        "_MiscFeature_TenC",
        "_GarageQual_Ex", "_PoolQC_Fa"
    ]
    proc_train_df.drop(drop_cols, axis=1, inplace=True)

    # Following columns are missing in train data so dropping it
    proc_test_df.drop(["_MSSubClass_150", "_Functional_None"], axis=1, inplace=True)

    drop_cols2 = [
        # only two are not zero
        "_Condition2_PosN",
        "_MSZoning_C (all)",
        "_MSSubClass_160",
    ]
    proc_train_df.drop(drop_cols2, axis=1, inplace=True)
    proc_test_df.drop(drop_cols2, axis=1, inplace=True)
    # Uncomment the following to run chosen model
    # xgb_pred = xgbr(proc_train_df, train_price_df, proc_test_df)
    # lasso_pred = lassor(proc_train_df, train_price_df, proc_test_df)
    # ridge_pred = ridger(proc_train_df, train_price_df, proc_test_df)
'''
    for i in range(len(xgb_pred)):
        print(np.exp((xgb_pred[i] + lasso_pred[i])/2))
    with open("XL_tunednewv6.csv", "w", newline='') as f:
        w1 = csv.writer(f)
        w1.writerow(["Id", "SalePrice"])
        for i in range(len(xgb_pred)):
            # w1.writerow([str(test_df["Id"][i]), str(np.exp((xgb_pred[i] + lasso_pred[i])/2))])
            w1.writerow([str(test_df["Id"][i]), str(np.exp((xgb_pred[i] + lasso_pred[i])/2))])
'''
