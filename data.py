from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import pandas
import numpy
import math

def transformEmbarked(x):
  if x == 'C': return 1
  elif x == 'Q': return 2
  else: return 3

def getTrainData():
  trainData = pandas.read_csv('./train.csv')
  trainOutcomes = trainData['Survived']
  trainData.drop(['Survived'], axis=1, inplace=True)
  return trainData, trainOutcomes

def getTestDataKaggle():
  testData = pandas.read_csv('./test.csv')
  passengerIds = testData['PassengerId']
  return testData, passengerIds

def dataLocal():
  trainData, trainOutcomes = getTrainData()
  trainData, avgAge, avgFare = transformData(trainData)
  trainData, testData, trainOutcomes, testOutcomes = train_test_split(trainData, trainOutcomes, test_size=0.15)
  return trainData, trainOutcomes, testData, testOutcomes

def dataKaggle():
  trainData, trainOutcomes = getTrainData()
  trainData, avgAge, avgFare = transformData(trainData)
  testData, passengerIds = getTestDataKaggle()
  testData, avgAge, avgFare = transformData(testData, avgAge, avgFare)
  return trainData, trainOutcomes, testData, passengerIds

def transformData(data, avgAge = None, avgFare = None):
  data['numRelatives'] = data['SibSp'] + data['Parch']
  data['hasRelatives'] = data['numRelatives'].apply(lambda x: 1 if x > 0 else 0)
  if avgAge == None:
    avgAge = numpy.nanmean(data['Age'])
  #TODO random?
  data['Age'] = data['Age'].apply(lambda x: avgAge if math.isnan(x) else x)
  data['isChild'] = data['Age'].apply(lambda x: 1 if x < 7 else 0)
  data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
  if avgFare == None:
    avgFare = numpy.nanmean(data['Fare'])
  #TODO random
  data['Fare'] = data['Fare'].apply(lambda x: avgFare if math.isnan(x) else x)
  embarked = pandas.get_dummies(data['Embarked'])
  embarked.drop(['Q'], axis=1, inplace=True)
  data = data.join(embarked)
  classes = pandas.get_dummies(data['Pclass'])
  classes.columns = ['Class1', 'Class2', 'Class3']
  classes.drop(['Class2'], axis=1, inplace=True)
  data = data.join(classes)
  data.drop(['Name', 'Ticket', 'PassengerId', 'numRelatives', 'SibSp', 'Parch', 'Cabin', 'Age', 'Embarked', 'Pclass'], axis=1, inplace=True)
  return data, avgAge, avgFare

def testLocal(model):
  trainData, trainOutcomes, testData, testOutcomes = dataLocal()
  model.fit(trainData, trainOutcomes)
  prediction = model.predict(testData)
  print(classification_report(testOutcomes, prediction))
  return prediction

def testKaggle(model):
  trainData, trainOutcomes, testData, passengerIds = dataKaggle()
  model.fit(trainData, trainOutcomes)
  prediction = model.predict(testData)
  predictionDF = pandas.DataFrame({
    'PassengerId': passengerIds,
    'Survived': prediction
  })
  predictionDF.to_csv('./prediction.csv', index=False)
  return prediction

if __name__ == '__main__':
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  testKaggle(RandomForestClassifier(n_estimators=100, n_jobs=2, max_depth=6))
  #testLocal(SVC(C=500, gamma=0.001))
