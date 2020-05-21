from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import NaiveBayes


class BN(object):
    def __init__(self, data):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        vectorizer = CountVectorizer(inputCol="words", outputCol="rawFeatures")

        idf = IDF(minDocFreq=3, inputCol="rawFeatures", outputCol="features")

        nb = NaiveBayes()

        pipeline = Pipeline(stages=[tokenizer, vectorizer, idf, nb])

        self.model = pipeline.fit(data)
