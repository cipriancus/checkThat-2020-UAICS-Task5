from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.classification import NaiveBayes


# The model is a Naive Bayes that uses the data provided by the organizers in 2020 plus some files from 2019 as training. The data is tokenized, converted to a vectors of token counts and finally  we compute the Inverse Document Frequency.
class BN(object):
    def __init__(self, data):
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        vectorizer = CountVectorizer(inputCol="words", outputCol="rawFeatures")

        idf = IDF(minDocFreq=3, inputCol="rawFeatures", outputCol="features")

        nb = NaiveBayes()

        pipeline = Pipeline(stages=[tokenizer, vectorizer, idf, nb])

        self.model = pipeline.fit(data)
