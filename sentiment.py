#! /usr/bin/python3

#----------Importing libraries--------------#

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SQLContext

from pyspark.sql.types import StructType
from pyspark.sql import functions as F


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
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
		df = sql_context.createDataFrame(spark_context.emptyRDD(), schema = StructType([]))
		for data in x:
			data = loads(data)
			df = sql_context.createDataFrame(data.values())
			#df = df.rdd.map(lambda x : (x[0] , x[1])).collect()
			#df = df.rdd
			
		return df.rdd
		
	except ValueError:
		print("Stopping stream job")
		ssc.stop(stopSparkContext=True, stopGraceFully=False)


## Function to preprocess tweets
def preprocessing(df):

	#df = print(df.map(lambda x : x.))
	#df = sql_context.createDataFrame(df, schema = ['feature0', 'feature1'])
	try:
		df = df.toDF()
		df = df.na.drop()
		
		# Defining regex patterns.
		urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
		userPattern = '@[^\s]+'
		alphaPattern = "[^a-zA-Z0-9]"
		sequencePattern = r"(.)\1\1+"
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
		#df = df.withColumn("feature1", F.regexp_replace(df['feature1'], sequencePattern, seqReplacePattern))
		
		#df = df.withColumn("feature1", F.regexp_replace(df['feature1'], "11", " "))
		# Replace more than one white space with a single white space
		df = df.withColumn("feature1", F.regexp_replace(df['feature1'], r"\s+", " "))
		
		return df.rdd
	
	except ValueError:
		print("Stopping stream job")
		ssc.stop(stopSparkContext=True, stopGraceFully=False)

def train(df):

	print("Entered Training")
	
	df = df.toDF()
	
	X_train = np.array([i[0] for i in df.select('feature1').collect()])
	Y_train = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	#print("Xtrain", X_train, type(X_train))
	#print("YTrain ", Y_train, type(Y_train))

	
	tokenizer_pkl = "./tokenizer.pkl"
	if ((os.path.exists(tokenizer_pkl)) and (os.path.getsize(tokenizer_pkl) != 0)):
		
		with open(tokenizer_pkl, 'rb') as f: 
			vectorizer = pickle.load(f)
		
		f.close()
		vectorizer.partial_fit(X_train)
		
		with open(tokenizer_pkl, 'ab') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	else:
		
		
		vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 12500, norm='l1')
		
		vectorizer.partial_fit(X_train)
		
		with open(tokenizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	
	
	X_train = vectorizer.fit_transform(X_train)
	
	
	print("Xtrain after :",X_train)
	#print("Ytrain after :",Y_train)
	#print("Type = ", type(X_train))
	
	print("DimX = ", X_train.ndim, "DimY", Y_train.ndim)
	print("ShapeX = ", X_train.shape, "ShapeY", Y_train.shape)

	
	naivebayes_pkl = "./MultinomialNB_2.pkl"
	
	if ((os.path.exists(naivebayes_pkl)) and (os.path.getsize(naivebayes_pkl) != 0)):
		
		with open(naivebayes_pkl, 'rb') as f: 
			naivebayes_model = pickle.load(f)
		
		f.close()
		naivebayes_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(naivebayes_pkl, 'ab') as f: 
			pickle.dump(naivebayes_model, f)
		
		f.close()
	else:
		
		#naivebayes_model = MultinomialNB()
		naivebayes_model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
		
		
		naivebayes_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(naivebayes_pkl, 'wb') as f: 
			pickle.dump(naivebayes_model, f)
		
		f.close()
		
	file_size = os.path.getsize(naivebayes_pkl)
	print("File Size is :", file_size, "bytes")
	
	return df.rdd
	

def test(df):
	
	print("Entered testing")
	print()
	
	df = df.toDF()
	X_test = np.array([i[0] for i in df.select('feature1').collect()])
	
	
	tokenizer_pkl = "./tokenizer.pkl"
	
	with open(tokenizer_pkl, 'rb') as f: 
		vectorizer = pickle.load(f)
		
	f.close()
	
	X_test = vectorizer.fit_transform(X_test)
	
	Y_test = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	naivebayes_pkl = "./MultinomialNB_2.pkl"
	
	with open(naivebayes_pkl, 'rb') as f: 
		naivebayes_model = pickle.load(f)
	
	file_size = os.path.getsize(naivebayes_pkl)
	print("File Size is :", file_size, "bytes")
	f.close()
	prediction = naivebayes_model.predict(X_test)
	print(prediction)
	
	print(naivebayes_model.score(X_test, Y_test))
	

	
	
#--------------------Main------------------------------#


if __name__ == '__main__':

	spark_context = SparkContext.getOrCreate()

	sql_context = SQLContext(spark_context)

	ssc = StreamingContext(spark_context, 12)

	datastream = ssc.socketTextStream("localhost",6100)

	data = datastream.transform(func)
	#data.pprint(25)

	processedDF = data.transform(preprocessing)
	#processedDF.pprint(25)

	#trainDF = processedDF.transform(train)
	#trainDF.pprint(2)

	#processedDF.foreachRDD(train)
	#processedDF.pprint(2)
	processedDF.foreachRDD(test)


	#testDF = processedDF.transform(test)
	#testDF.pprint(25)

	ssc.start()
	ssc.awaitTermination()

	#ssc.stop(stopGraceFully = True)
