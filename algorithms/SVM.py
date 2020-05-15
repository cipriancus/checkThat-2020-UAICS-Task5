from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF, Tokenizer, StopWordsRemover
from pyspark.ml.clustering import LDA


class SVM(object):
    def __init__(self, data):
        self.tokenizer = Tokenizer(inputCol="text", outputCol="rawWords")

        self.stopWords = StopWordsRemover(inputCol="rawWords", outputCol="words", caseSensitive=False,
                                          stopWords=StopWordsRemover.loadDefaultStopWords("english"))

        self.cv = CountVectorizer(inputCol="words", outputCol="rawFeatures")

        self.idf = IDF(inputCol="rawFeatures", outputCol="features")

        svm = LinearSVC()

        pipeline = Pipeline(stages=[self.tokenizer, self.stopWords, self.cv, self.idf, svm])

        self.model = pipeline.fit(data)