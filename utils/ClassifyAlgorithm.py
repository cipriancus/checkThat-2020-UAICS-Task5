from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def validate(obj, validate):
    prediction = obj.model.transform(validate)

    actual = [i.label for i in validate.select("label").collect()]

    predicted = [i.prediction for i in prediction.select("prediction").collect()]

    print(confusion_matrix(actual, predicted))
    print(classification_report(actual, predicted))


def classify(obj, sqlContext, classifier):
    from os import listdir
    import os

    schema = StructType([StructField('id', IntegerType(), False),
                         StructField('speaker', StringType(), False),
                         StructField('text', StringType(), False)])

    for file in listdir("."):
        test = sqlContext.read.format("com.databricks.spark.csv").options(header='false',
                                                                          inferschema='true',
                                                                          delimiter='\t').schema(
            schema).load(os.getcwd() + '/' + file)

        prediction = obj.model.transform(test)

        if(classifier != 'contrastive2'):
            selected = prediction.select("id", "text", "probability", "prediction")

            f = open('../results/' + classifier + '-' + file, "w+")

            for row in selected.collect():
                rid, text, prob, prediction = row
                f.write("%d\t%f\n" % (rid, prob[1]))
                # f.write("%d\t%f\t%f\t%s\n" % (rid, prob[1], prediction, text))

            f.close()
        else:
            selected = prediction.select("id", "text", "prediction")

            f = open('../results/' + classifier + '-' + file, "w+")

            for row in selected.collect():
                rid, text, prediction = row
                f.write("%d\t%f\n" % (rid, prediction))
                # f.write("%d\t%f\t%f\t%s\n" % (rid, prob[1], prediction, text))

            f.close()

