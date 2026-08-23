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
from sklearn.metrics import f1_score
from google.colab import drive

drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/SalesReinforcer/Src/')
import dataPrep

def splitBetweenTrainAndDev(df):
  train,dev = train_test_split(df,test_size=0.2,random_state=51)
  return train,dev

def trainAndTestModel(model, train, dev):
  model.fit(train.drop(columns=['isSubscribed']), train['isSubscribed'])
  train_predictions = model.predict(train.drop(columns=['isSubscribed']))
  dev_predictions = model.predict(dev.drop(columns=['isSubscribed']))
  trainScore = f1_score(train['isSubscribed'], train_predictions)
  testScore = f1_score(dev['isSubscribed'], dev_predictions)
  return trainScore, testScore

def createModel(criterion, nEstimators, maxDepth, maxFeatures):
  return rf(criterion=criterion, n_estimators=nEstimators, max_depth=maxDepth, max_features=maxFeatures)

def createSortedScoreVsHyperparameterDF(scoreVsHyperparameterDictionary):
  return pd.DataFrame(scoreVsHyperparameterDictionary).sort_values(by=['testScore','trainScore'],ascending=False)

def appendToHyperparameterDictionary(scoreVsHyperparameterDictionary, criterion, nEstimators, maxDepth, maxFeatures, trainScore, testScore):
  scoreVsHyperparameterDictionary['criterion'].append(criterion)
  scoreVsHyperparameterDictionary['nEstimators'].append(nEstimators)
  scoreVsHyperparameterDictionary['maxDepth'].append(maxDepth)
  scoreVsHyperparameterDictionary['maxFeatures'].append(maxFeatures)
  scoreVsHyperparameterDictionary['trainScore'].append(trainScore)
  scoreVsHyperparameterDictionary['testScore'].append(testScore)
  return scoreVsHyperparameterDictionary

def modelExistsInDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/Colab Notebooks/SalesReinforcer/Models/{filename}"
  return exists(fullName)

def saveModelToDrive(data,filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/Colab Notebooks/SalesReinforcer/Models/{filename}"
  pickle.dump(data, open(fullName, 'wb'))

def retrieveModelFromDrive(filename):
  # Drive mounts to /content/drive/My Drive/
  fullName = f"/content/drive/My Drive/Colab Notebooks/SalesReinforcer/Models/{filename}"
  return pickle.load(open(fullName, 'rb'))

def adjustRange(dfColumn):
  # Ensure min and max are called as methods on a pandas Series
  col_min = dfColumn.min()
  col_max = dfColumn.max()

  if col_min != col_max:
    return np.arange(col_min, col_max + 1, 1) # Include col_max in the range

  # Handle the case where all values are the same
  return np.arange(2, 2 * col_max, 1)




def main():
  train = dataPrep.retrieveCSVFromDrive("SalesReinforcerTrain.csv")
  test = dataPrep.retrieveCSVFromDrive("SalesReinforcerTest.csv")
  nColumns = len(train.columns)
  bestTestScore = 0

  possibleCriterion = ["gini","entropy","log_loss"]
  possibleNEstimators = np.arange(5, 100, 1)
  possibleMaxDepth = np.arange(2,80,1)
  possibleMaxFeatures = np.arange(2,120,1)

  scoreVsHyperparameterDictionary = {
      "criterion":[],
      "nEstimators":[],
      "maxDepth":[],
      "maxFeatures":[],
      "trainScore":[],
      "testScore":[]
  }

  for iteration in range(150):
    print(f"iteration: {iteration+1} of 150") # Updated print statement
    criterion = np.random.choice(possibleCriterion)
    nEstimators = np.random.choice(possibleNEstimators)
    maxDepth = np.random.choice(possibleMaxDepth)
    maxFeatures = np.random.choice(possibleMaxFeatures)
    model = createModel(criterion, nEstimators, maxDepth, maxFeatures)
    trainScore, testScore = trainAndTestModel(model, train, test)
    if (trainScore > .8) and (testScore > .8) and (testScore > bestTestScore):
      bestTestScore = testScore
      saveModelToDrive(model,"baseRandomForest.pkl")

    scoreVHyperparameterDictionary = appendToHyperparameterDictionary(scoreVsHyperparameterDictionary,
                                                                      criterion, nEstimators, maxDepth,
                                                                      maxFeatures, trainScore, testScore)
    if (iteration == 149):
      df = createSortedScoreVsHyperparameterDF(scoreVsHyperparameterDictionary)
      display(df)
      dataPrep.saveCSVToDrive(df,"BestHyperparameters.csv")
      if (modelExistsInDrive("baseRandomForest.pkl")):
        print('best model found')
      else:
        print('best model not found')
      return

    if (iteration % 10) == 9:
      bestFiveHyperparameterGroups = createSortedScoreVsHyperparameterDF(scoreVsHyperparameterDictionary).head(5)
      display(bestFiveHyperparameterGroups)
      possibleNEstimators = adjustRange(bestFiveHyperparameterGroups['nEstimators'])
      possibleMaxDepth = adjustRange(bestFiveHyperparameterGroups['maxDepth'])
      possibleMaxFeatures = adjustRange(bestFiveHyperparameterGroups['maxFeatures'])
      if (modelExistsInDrive("baseRandomForest.pkl")):
        print('best model found')
        testSet = dataPrep.retrieveCSVFromDrive("SalesReinforcerTest.csv")
        bestModel = retrieveModelFromDrive("baseRandomForest.pkl")
        test_predictions = bestModel.predict(testSet.drop(columns=['isSubscribed']))
        testScore = f1_score(testSet['isSubscribed'], test_predictions)
        print(f"test score: {testScore}")
        return
      scoreVsHyperparameterDictionary = {
          "criterion":[],
          "nEstimators":[],
          "maxDepth":[],
          "maxFeatures":[],
          "trainScore":[],
          "testScore":[]
      }

if __name__ == "__main__":
  main()
