import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
import pickle
from imblearn.over_sampling import SMOTENC
from os.path import exists
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.feature_selection import chi2
from scipy.stats import spearmanr
import json
from google.colab import drive

drive.mount('/content/drive')

def fileExistsInDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  return exists(fullName)

def savePickleFileToDrive(data,filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  pickle.dump(data, open(fullName, 'wb'))

def retrievePickleFileFromDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  return pickle.load(open(fullName, 'rb'))

def saveJSONToDrive(data,filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  with open(fullName, 'w') as fp:
      json.dump(data, fp)

def retrieveJSONFromDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  with open(fullName) as json_file:
      data = json.load(json_file)
  return data

def saveCSVToDrive(df,filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  df.to_csv(fullName,index=False)

def retrieveCSVFromDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/SalesReinforcerData/{filename}"
  return pd.read_csv(fullName)

def separateDFBySubtype(df,baseName):
  numericalCols = []
  categoricalCols = []
  for col in df.columns:
      if np.issubdtype(df[col].dtype, np.number):
          numericalCols.append(str(col))
      else:
          categoricalCols.append(str(col))
  numericalDF = df[numericalCols]
  categoricalDF = df[categoricalCols]
  savePickleFileToDrive(numericalCols, f"{baseName}NumericalCols.pkl")
  savePickleFileToDrive(categoricalCols, f"{baseName}CategoricalCols.pkl")
  return numericalDF,categoricalDF

def encodeTestDF(categoricalDF,baseName):
  ohe = retrievePickleFileFromDrive(f"{baseName}OneHotEncoder.pkl")
  le = retrievePickleFileFromDrive(f"{baseName}LabelEncoder.pkl")
  oheDF = ohe.transform(categoricalDF).fillna(0)
  leDF = le.transform(categoricalDF)
  return pd.concat([oheDF,leDF],axis=1)

def encodeDF(categoricalDF,baseName):
  ohe = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
  le = ce.OrdinalEncoder(return_df=True)
  oheDF = ohe.fit_transform(categoricalDF)
  oheColumns = list(oheDF.columns)
  savePickleFileToDrive(oheColumns,f"{baseName}OheColumns.pkl")
  leDF = le.fit_transform(categoricalDF)
  savePickleFileToDrive(ohe,f"{baseName}OneHotEncoder.pkl")
  savePickleFileToDrive(le,f"{baseName}LabelEncoder.pkl")
  return pd.concat([oheDF,leDF],axis=1)

def scaleTestDF(df,baseName):
  scaler = retrievePickleFileFromDrive(f"{baseName}Scaler.pkl")
  numericalCols = list(df.columns)
  # Use .loc for explicit assignment to prevent SettingWithCopyWarning
  df.loc[:, numericalCols] = scaler.transform(df.loc[:, numericalCols])
  return df

def scaleDF(df,baseName):
  scaler = StandardScaler()
  numericalCols = list(df.columns)
  # Use .loc for explicit assignment to prevent SettingWithCopyWarning
  df.loc[:, numericalCols] = scaler.fit_transform(df.loc[:, numericalCols])
  savePickleFileToDrive(scaler,f"{baseName}Scaler.pkl")
  return df

def processTestData(baseName):
  df = retrieveCSVFromDrive(f"{baseName}Test.csv")
  yArray = df[['isSubscribed']].values.ravel()
  df = df.drop('isSubscribed',axis = 1)

  rawDF = df.copy()
  saveCSVToDrive(rawDF,f"{baseName}RawTest.csv")

  numericalDF,categoricalDF = separateDFBySubtype(df,baseName)
  scaledDF = scaleTestDF(numericalDF,baseName)
  encodedDF = encodeTestDF(categoricalDF,baseName)
  finalDF = pd.concat([scaledDF,encodedDF],axis=1)
  finalDF['isSubscribed'] = yArray.reshape(-1,1)
  saveCSVToDrive(finalDF,f"{baseName}Test.csv")

def processTrainData(baseName):
  df = retrieveCSVFromDrive(f"{baseName}Train.csv")

  yArray = df[['isSubscribed']].values.ravel()
  df = df.drop('isSubscribed',axis = 1)

  rawDF = df.copy()
  saveCSVToDrive(rawDF,f"{baseName}RawTrain.csv")

  numericalDF,categoricalDF = separateDFBySubtype(df,baseName)
  scaledDF = scaleDF(numericalDF,baseName)
  encodedDF = encodeDF(categoricalDF,baseName)
  finalDF = pd.concat([scaledDF,encodedDF],axis=1)
  finalDF['isSubscribed'] = yArray.reshape(-1,1)
  saveCSVToDrive(finalDF,f"{baseName}Train.csv")


def transformDate(data):
  dateTimeColumns = ['First Contact','Last Contact', 'First Call', 'Signed up for a demo',
                     'Filled in customer survey','Did sign up to the platform','Account Manager assigned']
  for col in dateTimeColumns:
    data[col] = pd.to_datetime(data[col], format='%Y-%d-%m', errors='coerce')
    data[col + 'Year'] = data[col].dt.year
    data[col + 'Month'] = data[col].dt.month
    data[col + 'Day'] = data[col].dt.day
    data[col] = data[col].astype('int64')
  return data

def binarizeTargetsFromDF(df):
    df['Subscribed'] = pd.to_datetime(df['Subscribed'], format='%Y-%d-%m', errors='coerce').astype('int64')
    df['isSubscribed'] = (df['Subscribed'] > 0).astype(int)
    df.drop('Subscribed',axis=1,inplace=True)
    return df

def smote(X,y,baseName):
    # Create a copy to avoid modifying the original X directly
    X_temp = X.copy()

    # Separate numerical and categorical columns from the copy
    numerical_cols = X_temp.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_temp.select_dtypes(exclude=np.number).columns.tolist()

    # Apply Ordinal Encoding only for SMOTENC (temporary encoder)
    # handle_unknown='return_nan' means unknown categories will be encoded as NaN.
    temp_ordinal_encoder = ce.OrdinalEncoder(cols=categorical_cols, handle_unknown='return_nan')
    X_encoded_for_smote = temp_ordinal_encoder.fit_transform(X_temp)

    # Get the indices of categorical features in X_encoded_for_smote
    categorical_features_indices = [X_encoded_for_smote.columns.get_loc(col) for col in categorical_cols]

    # Initialize and apply SMOTENC
    balancer = SMOTENC(categorical_features=categorical_features_indices, random_state=51)
    resampledX_encoded_for_smote, resampledy = balancer.fit_resample(X_encoded_for_smote, y)

    # Inverse transform the categorical columns back to their original labels
    # The inverse_transform will handle the categorical columns, while numerical will remain as they were after SMOTENC.
    resampledX_original_format = temp_ordinal_encoder.inverse_transform(resampledX_encoded_for_smote)

    return resampledX_original_format, resampledy

def splitData(baseName):
    df = retrieveCSVFromDrive("cleanedData.csv")
    df = binarizeTargetsFromDF(df)
    originalY = df[['isSubscribed']]
    originalYArray = originalY.values.ravel()
    originalX = df.drop('isSubscribed',axis=1)

    # Split data first into training and testing sets
    XTrain, XTest, yTrain, yTest = train_test_split(originalX, originalYArray, test_size=0.2, random_state=51, stratify=originalYArray)

    # Apply SMOTE only to the training data
    resampledXTrain, resampledYTrain = smote(XTrain, yTrain, baseName)

    train = resampledXTrain.copy()
    train['isSubscribed'] = resampledYTrain
    saveCSVToDrive(train,f"{baseName}Train.csv")

    test = XTest.copy()
    test['isSubscribed'] = yTest
    saveCSVToDrive(test,f"{baseName}Test.csv")

def replaceNanWithZero(df):
  '''
  For Country and Education, sure it means we didn't get that info in, but for all the other columns,
  this is a status or date. No status or date, means said action never happened.
  e.g. person wasn't contacted, person did not subscribe, person wasn't assigned a manager,
  person is not in a stage or status etc...
  '''
  for col in df.columns:
    df[col] = df[col].fillna(0)
  return df

def main():
  data = retrieveCSVFromDrive("SalesCRM - CRM.csv")
  dateTransformedData = transformDate(data)
  nonNanData = replaceNanWithZero(dateTransformedData)
  baseName = "SalesReinforcer" # Define baseName here
  saveCSVToDrive(nonNanData,"cleanedData.csv")
  np.random.seed(51)
  # baseName = "SalesReinforcer" # Original line, now moved up
  if (fileExistsInDrive(f"{baseName}Train.csv") == False):
      splitData(baseName)
  processTrainData(baseName)
  processTestData(baseName)

if __name__ == "__main__":
    main()
