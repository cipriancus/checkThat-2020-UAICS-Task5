from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import LogisticRegression

class LR(object):

    def __init__(self, data):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")

        idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

        lr = LogisticRegression()

        pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

        self.model = pipeline.fit(data)


