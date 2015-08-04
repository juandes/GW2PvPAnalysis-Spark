from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel

conf = SparkConf().setMaster("spark://spark.master.url:7077").setAppName("GW2 PvP ML").set("spark.executor.memory", "1g")
sc = SparkContext(conf=conf)

# Load data and prepare it for the ML models
data = sc.textFile("team_result.txt")
data = data.map(lambda line: line.split(","))
data = data.map(lambda x: LabeledPoint(float(x[5]), [x[0], x[1], x[2], x[3], x[4]]))

# Split the dataset into training set (70%) and test set (30%)
trainingData, testData = data.randomSplit([0.7, 0.3], seed=1071)

# Create and train the naive Bayes model
naiveBayesModel = NaiveBayes.train(trainingData, 1.0)

# Apply the model to the test set
predictionAndLabelNaiveBayes = testData.map(lambda x: (naiveBayesModel.predict(x.features), x.label))

# Calculate the accuracy of the model
errorNaiveBayes = 1.0 * predictionAndLabelNaiveBayes.filter(lambda (x, y): x != y).count() / testData.count()
print "Naive Bayes model classification error: {0:f}".format(errorNaiveBayes)

# Create and train the random forest model
randomForestModel = RandomForest.trainClassifier(trainingData, numClasses=2,
                                                 categoricalFeaturesInfo={0: 9, 1: 9, 2: 9, 3: 9, 4: 9}, numTrees=3,
                                                 impurity="gini", maxDepth=4, maxBins=32, seed=1071)

'''
Note taken from the official API documentation:
In Python, predict cannot currently be used within an RDD
transformation or action. Call predict directly on the RDD instead.
'''
predictionsRandomForest = randomForestModel.predict(testData.map(lambda x: x.features))
labelsAndPredictionsRF = testData.map(lambda x: x.label).zip(predictionsRandomForest)
errorRandomForest = labelsAndPredictionsRF.filter(lambda (x, y): x != y).count() / float(testData.count())
print "Random forest classification error: {0:f}".format(errorRandomForest)