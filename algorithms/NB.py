from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer


class BN(object):

    def __init__(self, data):
        # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        # hashing tf mai prost decat vectorizer

        # hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
        vectorizer = CountVectorizer(inputCol="words", outputCol="rawFeatures")

        idf = IDF(minDocFreq=2, inputCol="rawFeatures", outputCol="features")

        # Naive Bayes model
        nb = NaiveBayes()

        # Pipeline Architecture
        pipeline = Pipeline(stages=[tokenizer, vectorizer, idf, nb])

        # Train model.  This also runs the indexers.
        self.model = pipeline.fit(data)
