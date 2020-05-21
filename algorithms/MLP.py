from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import MultilayerPerceptronClassifier


class MLP(object):

    def __init__(self, data):
        self.tokenizer = Tokenizer(inputCol="text", outputCol="words")

        self.hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)

        self.idf = IDF(inputCol="rawFeatures", outputCol="features")

        lr = MultilayerPerceptronClassifier(maxIter=1500, layers=[5000, 2500, 1000, 2], solver='gd')

        pipeline = Pipeline(stages=[self.tokenizer, self.hashingTF, self.idf, lr])

        self.model = pipeline.fit(data)
