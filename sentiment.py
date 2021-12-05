#! /usr/bin/python3

#----------Importing libraries--------------#

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import functions as F


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline

import numpy as np
from json import loads
import re
import pickle
import os

#--------------------Functions--------------------------#

## Function to create Dataframe 

def func(x):
	
	try:
		x = x.collect()
		print(len(x))
		df = sql_context.createDataFrame(spark_context.emptyRDD(), schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True)]))
		for data in x:
			data = loads(data)
			temp_df = sql_context.createDataFrame(data.values())
			#df = df.rdd.map(lambda x : (x[0] , x[1])).collect()
			#df = preprocessing(df)
			df = df.union(temp_df)
			
		df = preprocessing(df)
		print("DF length ", df.count())	
		return df.rdd
		
	except ValueError:
		print("Stopping stream job")
		ssc.stop(stopSparkContext=True, stopGraceFully=False)


## Function to preprocess tweets
def preprocessing(df):

	try:
		df = df.na.drop()
		
		# Defining regex patterns.
		urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
		userPattern = '@[^\s]+'
		alphaPattern = "[^a-zA-Z0-9]"
		sequencePattern = r"(.)\1+"
		seqReplacePattern = r"\1\1"
		
		# Replace all URls with 'URL'
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], urlPattern, " "))
		
		# Replace @USERNAME to 'USER'
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], userPattern, " "))
		
		# Replace all non alphabets.
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], alphaPattern, " "))
		
		# Replacing single characters
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], r"(?:^| )\w(?:$| )", " "))
		
		# Replace 3 or more consecutive letters by 2 letter
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'],  r'(.)\1{2,}', r'\1\1'))
		
		# Replace more than one white space with a single white space
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], r"\s+", " "))
		
		return df
	
	except ValueError:
		print("Stopping stream job")
		ssc.stop(stopSparkContext=True, stopGraceFully=False)

def train(df):

	print("Entered Training")
	
	df = df.toDF()
	print("DF length ", df.count())	
	
	X_train = np.array([i[0] for i in df.select('feature1').collect()])
	Y_train = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	#print("Xtrain", X_train, type(X_train))
	#print("YTrain ", Y_train, type(Y_train))

	
	vectorizer_pkl = "./Tokenizer/Hashing_Vectorizer.pkl"
	if ((os.path.exists(vectorizer_pkl)) and (os.path.getsize(vectorizer_pkl) != 0)):
		
		with open(vectorizer_pkl, 'rb') as f: 
			vectorizer = pickle.load(f)
		
		f.close()
		vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'ab') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	else:
		
		vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 1000000, norm='l1')
		vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	
	
	X_train = vectorizer.fit_transform(X_train)
	
	
	#print("Xtrain after :",X_train)
	#print("Ytrain after :",Y_train)
	#print("Type = ", type(X_train))
	
	print("DimX = ", X_train.ndim, "DimY", Y_train.ndim)
	print("ShapeX = ", X_train.shape, "ShapeY", Y_train.shape)

	
	model_pkl = "./Models/SVD_model.pkl"
	
	if ((os.path.exists(model_pkl)) and (os.path.getsize(model_pkl) != 0)):
		
		with open(model_pkl, 'rb') as f: 
			ml_model = pickle.load(f)
		
		f.close()
		ml_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(model_pkl, 'ab') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
	else:
	
		ml_model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=1000)
		ml_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(model_pkl, 'wb') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
		
	file_size = os.path.getsize(model_pkl)
	print("File Size is :", file_size, "bytes")
	
	return df.rdd
	

def test(df):
	
	print("Entered testing")
	
	df = df.toDF()
	X_test = np.array([i[0] for i in df.select('feature1').collect()])
	
	
	vectorizer_pkl = "./Tokenizer/Hashing_Vectorizer.pkl"
	
	with open(vectorizer_pkl, 'rb') as f: 
		vectorizer = pickle.load(f)
		
	f.close()
	
	X_test = vectorizer.fit_transform(X_test)
	
	Y_test = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	model_pkl = "./Models/SVD_model.pkl"
	
	with open(model_pkl, 'rb') as f: 
		ml_model = pickle.load(f)
	
	file_size = os.path.getsize(naivebayes_pkl)
	print("File Size is :", file_size, "bytes")
	f.close()
	prediction = ml_model.predict(X_test)
	print(prediction)
	
	
	print(ml_model.score(X_test, Y_test))
	

	
	
#--------------------Main------------------------------#


if __name__ == '__main__':

	spark_context = SparkContext.getOrCreate()

	sql_context = SQLContext(spark_context)

	ssc = StreamingContext(spark_context, 25)

	datastream = ssc.socketTextStream("localhost",6100)

	data = datastream.transform(func)
	data.pprint(25)
	
	data.foreachRDD(train)
	
	#data.pprint(10)
	#data.foreachRDD(test)
	
	
	#processedDF = data.transform(preprocessing)
	#processedDF.pprint(25)

	#trainDF = processedDF.transform(train)
	#trainDF.pprint(2)

	#processedDF.foreachRDD(train)
	#processedDF.pprint(2)
	#processedDF.foreachRDD(test)


	#testDF = processedDF.transform(test)
	#testDF.pprint(25)

	ssc.start()
	ssc.awaitTermination()

