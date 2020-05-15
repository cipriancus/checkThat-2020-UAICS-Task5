from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import MultilayerPerceptronClassifier


class MPClassifier(object):

    def __init__(self, data):
        self.tokenizer = Tokenizer(inputCol="text", outputCol="words")

        self.hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)

        self.idf = IDF(inputCol="rawFeatures", outputCol="features")

        lr = MultilayerPerceptronClassifier(maxIter=1500, layers=[5000, 1000, 2000, 2], blockSize=128, seed=1234)

        pipeline = Pipeline(stages=[self.tokenizer, self.hashingTF, self.idf, lr])

        self.model = pipeline.fit(data)
