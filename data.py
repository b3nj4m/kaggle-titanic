from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas
import numpy
import math
import matplotlib.pyplot as plot
import matplotlib
matplotlib.style.use('ggplot')

def plotSurvivedVsDied(data):
  survived = data[data['Survived'] == 1]
  died = data[data['Survived'] == 0]
  for col in data.columns:
    if data[col].dtype == numpy.int64 and data[col].dtype == numpy.float64:
      plot.figure()
      survived[col].plot.hist(alpha=0.5)
      died[col].plot.hist(alpha=0.5)
      plot.savefig('./survive-vs-die-' + col + '.png')
      plot.close()

def isNumeric(data, col):
  return data[col].dtype == numpy.int64 or data[col].dtype == numpy.float64

def plotPairwise(data):
  for col1 in data.columns:
    for col2 in data.columns:
      if not col1 == col2 and isNumeric(data, col1) and isNumeric(data, col2):
        plot.figure()
        data[[col1, col2]].plot.hist(alpha=0.5, stacked=True)
        plot.savefig('./' + col1 + '-vs-' + col2 + '.png')
        plot.close()

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
  trainData, avgAge, stdAge, avgFare, stdFare = transformData(trainData)
  trainData, testData, trainOutcomes, testOutcomes = train_test_split(trainData, trainOutcomes, test_size=0.15)
  return trainData, trainOutcomes, testData, testOutcomes

def dataKaggle():
  trainData, trainOutcomes = getTrainData()
  trainData, avgAge, stdAge, avgFare, stdFare = transformData(trainData)
  testData, passengerIds = getTestDataKaggle()
  testData, avgAge, stdAge, avgFare, stdFare = transformData(testData, avgAge, stdAge, avgFare, stdFare)
  return trainData, trainOutcomes, testData, passengerIds

def transformData(data, avgAge = None, stdAge = None, avgFare = None, stdFare = None):
  data['numRelatives'] = data['SibSp'] + data['Parch']
  data['hasRelatives'] = data['numRelatives'].apply(lambda x: 1 if x > 0 else 0)
  data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

  if avgAge == None:
    avgAge = numpy.nanmean(data['Age'])
  if stdAge == None:
    stdAge = numpy.nanstd(data['Age'])
  data['Age'] = data['Age'].apply(lambda x: numpy.random.normal(avgAge, stdAge) if math.isnan(x) else x)
  data['isChild'] = data['Age'].apply(lambda x: 1 if x < 10 else 0)

  if avgFare == None:
    avgFare = numpy.nanmean(data['Fare'])
  if stdFare == None:
    stdFare = numpy.nanstd(data['Fare'])
  data['Fare'] = data['Fare'].apply(lambda x: numpy.random.normal(avgFare, stdFare) if math.isnan(x) else x)
  data['hasHighFare'] = data['Fare'].apply(lambda x: 1 if x > avgFare else 0)

  embarked = pandas.get_dummies(data['Embarked'])
  embarked.drop(['Q'], axis=1, inplace=True)
  data = data.join(embarked)

  classes = pandas.get_dummies(data['Pclass'])
  classes.columns = ['isFirstClass', 'isSecondClass', 'isThirdClass']
  classes.drop(['isSecondClass'], axis=1, inplace=True)
  data = data.join(classes)

  data.drop(['Name', 'Ticket', 'PassengerId', 'numRelatives', 'SibSp', 'Parch', 'Cabin', 'Age', 'Embarked', 'Pclass', 'Fare'], axis=1, inplace=True)
  return data, avgAge, stdAge, avgFare, stdFare

def testLocal(model):
  trainData, trainOutcomes, testData, testOutcomes = dataLocal()
  model.fit(trainData, trainOutcomes)
  prediction = model.predict(testData)
  print(confusion_matrix(testOutcomes, prediction))
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
  from sklearn.neighbors import KNeighborsClassifier
  testKaggle(KNeighborsClassifier(n_neighbors=40, n_jobs=2, weights='distance'))
  #testLocal(RandomForestClassifier(n_estimators=100, n_jobs=2, max_depth=6))
  #testLocal(SVC(C=500, gamma=0.001))
