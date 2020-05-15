from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def read_dataset(sqlContext):
    from os import listdir

    firstIteration = True

    data = []

    schema = StructType([StructField('id', IntegerType(), False),
                         StructField('speaker', StringType(), False),
                         StructField('text', StringType(), False),
                         StructField('label', IntegerType(), False)])

    for file in listdir("."):
        if firstIteration == True:
            data = sqlContext.read.format("com.databricks.spark.csv").options(header='false',
                                                                              inferschema='true',
                                                                              delimiter='\t').schema(
                schema).load(file)
            firstIteration = False
        else:
            data = data.union(
                sqlContext.read.format("com.databricks.spark.csv").options(header='false', inferschema='true',
                                                                           delimiter='\t').schema(
                    schema).load(file))
    return data
