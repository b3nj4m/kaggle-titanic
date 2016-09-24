from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy
import math

#TODO pull transforms/features out into separate lib
def transformEmbarked(x):
  if x == 'C': return 1
  elif x == 'Q': return 2
  else: return 3

trainData = pandas.read_csv('./train.csv')
trainOutcomes = trainData['Survived']
#TODO decide whether there's anything important in the names
trainData['numRelatives'] = trainData['SibSp'] + trainData['Parch']
trainData['hasRelatives'] = trainData['numRelatives'].apply(lambda x: 1 if x > 0 else 0)
avgAge = numpy.nanmean(trainData['Age'])
trainData['Age'] = trainData['Age'].apply(lambda x: avgAge if math.isnan(x) else x)
trainData['isChild'] = trainData['Age'].apply(lambda x: 1 if x < 7 else 0)
trainData = trainData[trainData.columns.difference(['Survived', 'Name', 'Ticket', 'PassengerId', 'numRelatives', 'SibSp', 'Parch', 'Cabin', 'Age'])]
trainData['Sex'] = trainData['Sex'].apply(lambda x: 1 if x == 'male' else 0)
avgFare = numpy.nanmean(trainData['Fare'])
trainData['Fare'] = trainData['Fare'].apply(lambda x: avgFare if math.isnan(x) else x)
trainData['Embarked'] = trainData['Embarked'].apply(transformEmbarked)
#trainData['Cabin'] = trainData['Cabin'].apply(lambda x: 1 if isinstance(x, str) else 0)

testData = pandas.read_csv('./test.csv')
passengerIds = testData['PassengerId']
testData['numRelatives'] = testData['SibSp'] + testData['Parch']
testData['hasRelatives'] = testData['numRelatives'].apply(lambda x: 1 if x > 0 else 0)
testData['Age'] = testData['Age'].apply(lambda x: avgAge if math.isnan(x) else x)
testData['isChild'] = testData['Age'].apply(lambda x: 1 if x < 7 else 0)
testData = testData[testData.columns.difference(['Name', 'Ticket', 'PassengerId', 'numRelatives', 'SibSp', 'Parch', 'Cabin', 'Age'])]
testData['Sex'] = testData['Sex'].apply(lambda x: 1 if x == 'male' else 0)
testData['Fare'] = testData['Fare'].apply(lambda x: avgFare if math.isnan(x) else x)
testData['Embarked'] = testData['Embarked'].apply(transformEmbarked)
#testData['Cabin'] = testData['Cabin'].apply(lambda x: 1 if isinstance(x, str) else 0)

model = RandomForestClassifier(max_depth=8, n_jobs=2, n_estimators=50)
model.fit(trainData, trainOutcomes)

prediction = model.predict(testData)

predictionDF = pandas.DataFrame()
predictionDF['PassengerId'] = passengerIds
predictionDF['Survived'] = prediction
predictionDF.to_csv('./rf-prediction.csv', index=False)
