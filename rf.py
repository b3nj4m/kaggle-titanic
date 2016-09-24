from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy
import math

def transformEmbarked(x):
  if x == 'C': return 1
  elif x == 'Q': return 2
  else: return 3

trainData = pandas.read_csv('./train.csv')
trainOutcomes = trainData['Survived']
#TODO decide whether embarked is important
#TODO decide whether there's anything important in the names
#TODO decide whether to combine/partition children/sibling counts (could combine into a relatives count, could create yes/no for > N relatives, ...)
trainData = trainData[trainData.columns.difference(['Survived', 'Cabin', 'Name', 'Ticket', 'PassengerId'])]
trainData['Sex'] = trainData['Sex'].apply(lambda x: 1 if x == 'male' else 0)
avgAge = numpy.nanmean(trainData['Age'])
trainData['Age'] = trainData['Age'].apply(lambda x: avgAge if math.isnan(x) else x)
avgFare = numpy.nanmean(trainData['Fare'])
trainData['Fare'] = trainData['Fare'].apply(lambda x: avgFare if math.isnan(x) else x)
trainData['Embarked'] = trainData['Embarked'].apply(transformEmbarked)

testData = pandas.read_csv('./test.csv')
passengerIds = testData['PassengerId']
testData = testData[testData.columns.difference(['Cabin', 'Name', 'Ticket', 'PassengerId'])]
testData['Sex'] = testData['Sex'].apply(lambda x: 1 if x == 'male' else 0)
testData['Age'] = testData['Age'].apply(lambda x: avgAge if math.isnan(x) else x)
testData['Fare'] = testData['Fare'].apply(lambda x: avgFare if math.isnan(x) else x)
testData['Embarked'] = testData['Embarked'].apply(transformEmbarked)

model = RandomForestClassifier(max_depth=6)
model.fit(trainData, trainOutcomes)

prediction = model.predict(testData)

predictionDF = pandas.DataFrame()
predictionDF['PassengerId'] = passengerIds
predictionDF['Survived'] = prediction
predictionDF.to_csv('./rf-prediction.csv', index=False)
