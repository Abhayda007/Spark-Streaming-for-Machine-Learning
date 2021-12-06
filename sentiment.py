#! /usr/bin/python3

#----------Importing libraries--------------#

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import functions as F


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

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
		
		df = sql_context.createDataFrame(spark_context.emptyRDD(), schema = StructType([StructField("feature0", StringType(), True), StructField("feature1", StringType(), True)]))
		for data in x:
			data = loads(data)
			temp_df = sql_context.createDataFrame(data.values())
			
			df = df.union(temp_df)
			
		df = preprocessing(df)
		
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
		df = df.withColumn("feature1", F.regexp_replace("feature1", urlPattern, " "))
		
		# Replace @USERNAME to 'USER'
		df = df.withColumn("feature1", F.regexp_replace("feature1", userPattern, " "))
		
		# Replace all non alphabets.
		df = df.withColumn("feature1", F.regexp_replace("feature1", alphaPattern, " "))
		
		#converting to lower case
		df = df.withColumn("feature1", F.lower(df['feature1']))
		
		# Replace 3 or more consecutive letters by 2 letter
		df = df.withColumn("feature1", F.regexp_replace("feature1",  r'[^\w\s]|(.)(?=\1\1)', ''))
		
		# Replacing single characters
		df = df.withColumn("feature1", F.regexp_replace("feature1", r"\b\w\b", ""))
		
		# Replace more than one white space with a single white space
		df = df.withColumn("feature1", F.regexp_replace("feature1", r"\s+", " "))
		
		return df
	
	except ValueError:
		print("Stopping stream job")
		ssc.stop(stopSparkContext=True, stopGraceFully=False)

def train(df):

	print("Entered Training")
	
	df = df.toDF()
	
	X_train = np.array([i[0] for i in df.select('feature1').collect()])
	Y_train = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	vectorizer_pkl = "./Vectorizer/PAC_Hashing_Vectorizer.pkl"
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	
	if ((os.path.exists(vectorizer_pkl)) and (os.path.getsize(vectorizer_pkl) != 0)):
		
		with open(vectorizer_pkl, 'rb') as f: 
			vectorizer = pickle.load(f)
		
		f.close()
		vectorizer = vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	else:
		
		vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 1000000, norm='l1')
		vectorizer = vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	
	X_train = vectorizer.fit_transform(X_train)
	
	#model_pkl = "./Models/SVD_model.pkl"
	model_pkl = "./Models/PAC_model.pkl"
	#model_pkl = "./Models/PTRON_model.pkl"
	#model_pkl = "./Models/BernoulliNB_model.pkl"
	
	if ((os.path.exists(model_pkl)) and (os.path.getsize(model_pkl) != 0)):
		
		with open(model_pkl, 'rb') as f: 
			ml_model = pickle.load(f)
		
		f.close()
		ml_model = ml_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(model_pkl, 'wb') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
	else:
	
		#ml_model = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, random_state=10, max_iter=1000)
		ml_model = PassiveAggressiveClassifier(random_state = 10 )
		#ml_model = Perceptron(n_iter=100 ,random_state=10)
		#ml_model = BernoulliNB()
		
		ml_model = ml_model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
		
		with open(model_pkl, 'wb') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
			
	return df.rdd
	

def test(df):
	
	print("Entered testing")
	
	df = df.toDF()
	X_test = np.array([i[0] for i in df.select('feature1').collect()])
	
	
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	vectorizer_pkl = "./Vectorizer/PAC_Hashing_Vectorizer.pkl"
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	#vectorizer_pkl = "./Vectorizer/Hashing_Vectorizer.pkl"
	
	with open(vectorizer_pkl, 'rb') as f: 
		vectorizer = pickle.load(f)
		
	f.close()
	
	#vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 1000000, norm='l1')
	X_test = vectorizer.fit_transform(X_test)
	
	Y_test = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	#model_pkl = "./Models/SVD_model.pkl"
	model_pkl = "./Models/PAC_model.pkl"
	#model_pkl = "./Models/PTRON_model.pkl"
	#model_pkl = "./Models/BernoulliNB_model.pkl"
	
	with open(model_pkl, 'rb') as f: 
		ml_model = pickle.load(f)
	
	f.close()
	
	prediction = ml_model.predict(X_test)
	
	print("Accuracy : ", accuracy_score(Y_test, prediction) * 100)
	print("Classification_report : ", classification_report(Y_test, prediction))
	

def clustering(df):
	
	print("Entered Clustering")
	
	df = df.toDF()
	
	X_train = np.array([i[0] for i in df.select('feature1').collect()])
	Y_train = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	
	vectorizer_pkl = "./Vectorizer/Clustering_Vectorizer.pkl"
	if ((os.path.exists(vectorizer_pkl)) and (os.path.getsize(vectorizer_pkl) != 0)):
		
		with open(vectorizer_pkl, 'rb') as f: 
			vectorizer = pickle.load(f)
		
		f.close()
		vectorizer = vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	else:
		
		vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 1000000, norm='l1')
		vectorizer = vectorizer.partial_fit(X_train)
		
		with open(vectorizer_pkl, 'wb') as f: 
			pickle.dump(vectorizer, f)
		
		f.close()
	
	
	X_train = vectorizer.fit_transform(X_train)
	
	print("DimX = ", X_train.ndim, "DimY", Y_train.ndim)
	print("ShapeX = ", X_train.shape, "ShapeY", Y_train.shape)
	
	model_pkl = "./Models/Clustering_model.pkl"

	
	if ((os.path.exists(model_pkl)) and (os.path.getsize(model_pkl) != 0)):
		
		with open(model_pkl, 'rb') as f: 
			ml_model = pickle.load(f)
		
		f.close()
		ml_model = ml_model.partial_fit(X_train)
		
		with open(model_pkl, 'wb') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
	else:
	
		ml_model = MiniBatchKMeans(n_clusters = 2, batch_size = 100, random_state = 10, max_iter=100)
		ml_model = ml_model.partial_fit(X_train)
		
		with open(model_pkl, 'wb') as f: 
			pickle.dump(ml_model, f)
		
		f.close()
	
	return df.rdd
	
def clustering_test(df):
	
	print("Entered testing")
	
	df = df.toDF()
	X_test = np.array([i[0] for i in df.select('feature1').collect()])
	
	
	vectorizer_pkl = "./Vectorizer/Clustering_Vectorizer.pkl"
	
	with open(vectorizer_pkl, 'rb') as f: 
		vectorizer = pickle.load(f)
		
	f.close()	
	
	#vectorizer = HashingVectorizer(analyzer='word' , stop_words='english', n_features = 1000000, norm='l1')
	X_test = vectorizer.fit_transform(X_test)
	
	Y_test = np.array([int(i[0]) for i in df.select('feature0').collect()])
	
	model_pkl = "./Models/Clustering_model.pkl"
	
	with open(model_pkl, 'rb') as f: 
		ml_model = pickle.load(f)

	f.close()
	prediction = ml_model.predict(X_test)

	ar_unique, i = np.unique(prediction, return_counts=True)
	
	print("Unique values:", ar_unique)
	print("Counts:", i)
	
	print("Accuracy : ", accuracy_score(Y_test, prediction) * 100)
	print("Classification_report : ", classification_report(Y_test, prediction))


#---------------------------Main------------------------------#


if __name__ == '__main__':

	spark_context = SparkContext.getOrCreate()

	sql_context = SQLContext(spark_context)

	ssc = StreamingContext(spark_context, 30)

	datastream = ssc.socketTextStream("localhost",6100)

	data = datastream.transform(func)
	data.pprint(10)
	
	#Classification
	
	#data.foreachRDD(train)
	data.foreachRDD(test)
	
	#Clustering
	
	#data.foreachRDD(clustering)
	#data.foreachRDD(clustering_test)
	
	ssc.start()
	ssc.awaitTermination()

