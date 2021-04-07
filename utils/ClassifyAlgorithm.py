from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from os import listdir
import os

from .task5 import calc_metrics


def validate(obj, sqlContext, classifier):
    os.chdir("../validate")

    schema = StructType([StructField('id', IntegerType(), False),
                         StructField('speaker', StringType(), False),
                         StructField('text', StringType(), False),
                         StructField('label', IntegerType(), False)])

    validate_files = ""
    gold_files = ""

    for file in listdir("."):
        if not os.path.isdir(os.getcwd() + '/' + file):

            test = sqlContext.read.format("com.databricks.spark.csv").options(header='false',
                                                                              inferschema='true',
                                                                              delimiter='\t').schema(
                schema).load(os.getcwd() + '/' + file)

            actual = [i.label for i in test.select("label").collect()]

            prediction = obj.model.transform(test)

            selected = prediction.select("id", "text", "probability", "prediction")

            f = open('./results/' + classifier + '-' + file, "w+")

            g = open('./results/gold-' + classifier + '-' + file, "w+")

            iterator = 0

            for row in selected.collect():
                rid, text, prob, prediction = row
                f.write("%d\t%f\n" % (rid, prob[1]))
                g.write("%d\t\t\t%d\n" % (rid, int(actual[iterator])))
                iterator = iterator + 1
            validate_files = validate_files + os.getcwd() + '/results/' + classifier + '-' + file + ","
            gold_files = gold_files + os.getcwd() + '/results/gold-' + classifier + '-' + file + ","
            f.close()
            g.close()

    validate_files = validate_files[:-1]
    gold_files = gold_files[:-1]
    calc_metrics(validate_files, gold_files)


def classify(obj, sqlContext, classifier):
    os.chdir("../test")

    schema = StructType([StructField('id', IntegerType(), False),
                         StructField('speaker', StringType(), False),
                         StructField('text', StringType(), False)])

    for file in listdir("."):
        if not os.path.isdir(os.getcwd() + '/' + file):

            test = sqlContext.read.format("com.databricks.spark.csv").options(header='false',
                                                                              inferschema='true',
                                                                              delimiter='\t').schema(
                schema).load(os.getcwd() + '/' + file)

            prediction = obj.model.transform(test)

            selected = prediction.select("id", "text", "probability", "prediction")

            f = open('./results/' + classifier + '/' + file, "w+")

            for row in selected.collect():
                rid, text, prob, prediction = row
                f.write("%d\t%f\n" % (rid, prob[1]))
            f.close()
