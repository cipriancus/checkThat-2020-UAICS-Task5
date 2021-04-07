from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from algorithms import LR as lr
from algorithms import DT as dt
from algorithms import NB as bn
import utils.BuildDataset as buildDataset
import utils.ClassifyAlgorithm as clsf
import warnings
import os

os.chdir("data/training")

warnings.filterwarnings("ignore")

conf = SparkConf().setAppName("clef")

conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '45G')
        .set('spark.driver.memory', '45G')
        .set('spark.driver.maxResultSize', '45G'))

spark_context = SparkContext(conf=conf)

sqlContext = SQLContext(spark_context)

train = buildDataset.read_dataset(sqlContext)

print('---- 1 -----')
print('Bayes Naive')
bn_classifier = bn.BN(train)
#clsf.validate(bn_classifier, sqlContext, 'primary')
clsf.classify(bn_classifier, sqlContext, 'primary')
del bn_classifier

print('---- 2 -----')
print('Logistic Regression')
lr_classifier = lr.LR(train)
#clsf.validate(lr_classifier, sqlContext, 'contrastive-1')
clsf.classify(lr_classifier, sqlContext, 'contrastive-1')
del lr_classifier

print('---- 3 -----')
print('DT')
dt_alg = dt.DT(train)
#clsf.validate(dt_alg, sqlContext, 'contrastive-2')
clsf.classify(dt_alg, sqlContext, 'contrastive-2')
del dt_alg

