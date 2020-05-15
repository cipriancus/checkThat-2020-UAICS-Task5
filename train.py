from pyspark import SparkContext, SparkConf
import utils.ClassifyAlgorithm as clsf
from algorithms import MultilayerPerceptron as multiLayPer
from algorithms import LogisticRegression as lr
from algorithms import SVM as svm
from algorithms import NB as bn
import warnings
import os
import utils.BuildDataset as buildDataset
from pyspark.sql import SQLContext

os.chdir("./data/2019/training")

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '12G')
        .set('spark.driver.memory', '12G'))

spark_context = SparkContext(conf=conf)

sqlContext = SQLContext(spark_context)

data = buildDataset.read_dataset(sqlContext)

train, validate = data.randomSplit([1.0, 0.0], seed=12345)

os.chdir("../test")

print(os.getcwd())

print('---- 1 -----')
print('Bayes Naive')
bn_classifier = bn.BN(train)
print(clsf.validate(bn_classifier, validate))
print(clsf.classify(bn_classifier, sqlContext, 'primary'))

# del bn_classifier
#
# print('---- 2 -----')
# print('Logistic Regression')
# lr_classifier = lr.LR(train)
# #print(clsf.validate(lr_classifier, validate))
# print(clsf.classify(lr_classifier, sqlContext, 'contrastive1'))
#
# del lr_classifier
#
# print('---- 3 -----')
# print('SVM')
# svm_class = svm.SVM(train)
# #print(clsf.validate(svm_class, validate))
# print(clsf.classify(svm_class, sqlContext,'contrastive2'))
#
# del svm_class

# print('---- 5 -----')
# print('Multi Layer Perceptron Neural Network')
# multiLayPer_classifier = multiLayPer.MPClassifier(train)
# print(clsf.validate(multiLayPer_classifier, validate))
# print(clsf.classify(multiLayPer_classifier, sqlContext,'mlp'))
