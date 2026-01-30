#spark [[Spark]]

* SparkContext represents the connection to a Spark cluster.
	* It acts as a coordinator and is responsible for task execution across the cluster.
	* It used to be the entry point for applications in earlier versions of Spark (1.x)
	* Provides core functionality for low level programming and cluster interaction. 
* SparkSession replaces SparkContext as the entry point
	* Combines SparkContext, SQLContext, HiveContext, StreamingContext.
* 