from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import DecisionTreeClassifier


class DT(object):
    def __init__(self, data):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        vectorizer = CountVectorizer(inputCol="words", outputCol="rawFeatures")

        idf = IDF(minDocFreq=3, inputCol="rawFeatures", outputCol="features")

        dt = DecisionTreeClassifier(maxDepth=30, maxBins=128, minInstancesPerNode=5, maxMemoryInMB=4096)

        pipeline = Pipeline(stages=[tokenizer, vectorizer, idf, dt])

        self.model = pipeline.fit(data)
