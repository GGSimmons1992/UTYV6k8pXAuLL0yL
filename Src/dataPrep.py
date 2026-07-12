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

def removeUnimportantCategoricalColumns(categoricalDF,y,combinedName,datasetType = 'train'):
  if datasetType == 'test':
      significantColumns = retrievePickleFileFromDrive(f"{combinedName}SignificantCategoricalCols.pkl")
  else:
      le = ce.OrdinalEncoder(return_df=True)
      leDF = le.fit_transform(categoricalDF)
      pValues = chi2(leDF, y)[1]
      pValueDF = pd.DataFrame({"feature":list(categoricalDF.columns),"pValue":pValues},columns=["feature","pValue"],index=None)
      lowPDF = pValueDF[pValueDF["pValue"] < 0.05]
      significantColumns = list(lowPDF['feature'])
      savePickleFileToDrive(significantColumns, f"{combinedName}SignificantCategoricalCols.pkl")
  return categoricalDF[significantColumns]

def removeUnimportantNumericalColumns(numericalDF,y,combinedName,datasetType = 'train'):
    if datasetType == 'test':
        significantColumns = retrievePickleFileFromDrive(f"{combinedName}SignificantNumericalCols.pkl")
    else:
        numericalCols = list(numericalDF.columns)
        significantColumns = []
        allNumericalColumns = numericalDF.columns
        for col in allNumericalColumns:
            x = numericalDF[[col]].values.ravel()
            p = spearmanr(x,y)[1]
            if p < 0.05:
                significantColumns.append(str(col))
        savePickleFileToDrive(significantColumns, f"{combinedName}SignificantNumericalCols.pkl")
    return numericalDF[significantColumns]

def encodeTestDF(categoricalDF,baseName):
  ohe = retrievePickleFileFromDrive(f"{baseName}OneHotEncoder.pkl")
  le = retrievePickleFileFromDrive(f"{baseName}LabelEncoder.pkl")
  oheDF = ohe.transform(categoricalDF).fillna(0)
  leDF = le.transform(categoricalDF)
  return pd.concat([oheDF,leDF],axis=1)

def encodeDF(categoricalDF,baseName):
  ohe = ce.OneHotEncoder(handle_unknown='ignore',return_df=True,use_cat_names=True)
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
  df[numericalCols] = scaler.transform(df[numericalCols])
  return df

def scaleDF(df,baseName):
  scaler = StandardScaler()
  numericalCols = list(df.columns)
  df[numericalCols] = scaler.fit_transform(df[numericalCols])
  savePickleFileToDrive(scaler,f"{baseName}Scaler.pkl")
  return df

def processTestData(baseName):
  df = retrieveCSVFromDrive(f"{baseName}Test.csv")
  yArray = df[['isSubscribed']].values.ravel()
  df = df.drop('isSubscribed',axis = 1)
  numericalDF = removeUnimportantNumericalColumns(df,yArray,baseName,"test")
  categoricalDF = removeUnimportantCategoricalColumns(df,yArray,baseName,"test")
  scaledDF = scaleTestDF(numericalDF,baseName)
  encodedDF = encodeTestDF(categoricalDF,baseName)
  finalDF = pd.concat([scaledDF,encodedDF],axis=1)
  finalDF['isSubscribed'] = yArray.reshape(-1,1)
  saveCSVToDrive(finalDF,f"{baseName}Test.csv")

def processTrainData(baseName):
  df = retrieveCSVFromDrive(f"{baseName}Train.csv")
  print(df.columns)
  yArray = df[['isSubscribed']].values.ravel()
  df = df.drop('isSubscribed',axis = 1)
  numericalDF,categoricalDF = separateDFBySubtype(df,baseName)
  numericalDF = removeUnimportantNumericalColumns(numericalDF,yArray,baseName)
  categoricalDF = removeUnimportantCategoricalColumns(categoricalDF,yArray,baseName)
  scaledDF = scaleDF(numericalDF,baseName)
  encodedDF = encodeDF(categoricalDF,baseName)
  finalDF = pd.concat([scaledDF,encodedDF],axis=1)
  finalDF['isSubscribed'] = yArray.reshape(-1,1)
  saveCSVToDrive(finalDF,f"{baseName}Train.csv")


def transformDate(data):
  dateTimeColumns = ['First Contact','Last Contact', 'First Call', 'Signed up for a demo',
                     'Filled in customer survey','Did sign up to the platform','Account Manager assigned','Subscribed']
  for col in dateTimeColumns:
    data[col] = pd.to_datetime(data[col], format='%Y-%d-%m', errors='coerce')
    data[col + 'Year'] = data[col].dt.year
    data[col + 'Month'] = data[col].dt.month
    data[col + 'Day'] = data[col].dt.day
    data[col] = data[col].astype('int64')
  return data

def binarizeTargetsFromDF(df):
    df['isSubscribed'] = (df['Subscribed'] > 0).astype(int)
    return df

def smote(X,y,baseName):
    categoricalVariables = [i for i,col in enumerate(X.columns) if not np.issubdtype(X[col].dtype, np.number)]
    balancer = SMOTENC(categorical_features=categoricalVariables,random_state=51)
    categoricalVariables = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
    encoder = ce.OneHotEncoder(cols=categoricalVariables)
    X_encoded = encoder.fit_transform(X)
    resampledX,resampledy = balancer.fit_resample(X_encoded, y)
    return encoder.inverse_transform(resampledX),resampledy

def splitData(baseName):
    df = retrieveCSVFromDrive("cleanedData.csv")
    df = binarizeTargetsFromDF(df)
    originalY = df[['isSubscribed']]
    originalYArray = originalY.values.ravel()
    originalX = df.drop('isSubscribed',axis=1)
    X,y =smote(originalX,originalYArray,baseName)
    XTrain,XTest,yTrain,yTest = train_test_split(X, y, test_size=0.2,random_state=51)
    train = XTrain.copy()
    train['isSubscribed'] = yTrain
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
