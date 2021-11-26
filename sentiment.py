#! /usr/bin/python3

#----------Importing libraries--------------#

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SQLContext

from pyspark.sql.types import StructType
from pyspark.sql import functions as F


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import numpy as np
from json import loads
import re
import pickle
import os

#-------------Initialization-------------------#

spark_context = SparkContext.getOrCreate()

sql_context = SQLContext(spark_context)

ssc = StreamingContext(spark_context, 5)

datastream = ssc.socketTextStream("localhost",6100)

#---------------------Accumulator Variables--------------------#




#--------------------Functions--------------------------#

## Function to create Dataframe 

def func(x):
	x = x.collect()
	df = sql_context.createDataFrame(spark_context.emptyRDD(), schema = StructType([]))
	for data in x:
		data = loads(data)
		df = sql_context.createDataFrame(data.values())
		#df = df.rdd.map(lambda x : (x[0] , x[1])).collect()
		df = df.rdd
		
	return df


## Function to preprocess tweets
def preprocessing(df):

	#df = print(df.map(lambda x : x.))
	#df = sql_context.createDataFrame(df, schema = ['feature0', 'feature1'])
	
	df = df.toDF()
	df = df.na.drop()

	# Defining regex patterns.
	urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
	userPattern = '@[^\s]+'
	alphaPattern = "[^a-zA-Z0-9_]"
	sequencePattern = r"(.)\1\1+"
	seqReplacePattern = r"\1\1"
	
	# Replace all URls with 'URL'
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], urlPattern, ' URL'))
	# Replace @USERNAME to 'USER'
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], userPattern, ' USER'))
	# Replace all non alphabets.
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], alphaPattern, " "))
	# Replacing single characters
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], r"(?:^| )\w(?:$| )", " "))
	# Replace 3 or more consecutive letters by 2 letter
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], sequencePattern, seqReplacePattern))
	
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], "11", " "))
	# Replace more than one white space with a single white space
	df = df.withColumn("feature1", F.regexp_replace(df['feature1'], r"\s+", " "))
	
	return df.rdd

def train(df):

	print("Entered Training")
	
	df = df.toDF()
	
	X_train = np.array([i[0] for i in df.select('feature1').collect()])
	Y_train = np.array([i[0] for i in df.select('feature0').collect()])
	
	#print("Xtrain", X_train, type(X_train))
	#print("YTrain ", Y_train, type(Y_train))
	
	vectorizer = TfidfVectorizer()
	X_train = vectorizer.fit_transform(X_train)
	
	print(X_train)
	print("Type = ", type(X_train))
	
	trained_model = MultinomialNB().partial_fit(X_train, Y_train, classes=np.unique(Y_train))
	
	print(type(trained_model))
	print(trained_model.get_params())
	
	return df.rdd

	
	
#--------------------Main------------------------------#


data = datastream.transform(func)
#data.pprint(50)


processedDF = data.transform(preprocessing)
#processedDF.pprint(50)


trainDF = processedDF.transform(train)
trainDF.pprint(25)

ssc.start()
ssc.awaitTermination()


#ssc.stop()
