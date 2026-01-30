#spark 

* Features:
	* Designed for large scale distributed data processing.
	* Provides in memory storage for intermediate computations.
	* Provides fault tolerance and reliability.
	* Due to caching of frequently used data and in-memory computations, Spark is a useful tool for iterative algos as well as interactive data analysis.
* Components 
	* [[Spark Core]]
		* Converts the code into DAG.
		* The code is converted to byte code for the JVM workers to work with.
		* Because of this bytecode it doesn't matter which language you are coding in.
	* [[MLib]]
		* Contains features for creating features, build pipelines and persisting/loading models.
		* Also includes math and stats libraries.
	* [[Spark SQL]]
		* Supports multiple formats such as RDBMSs, Avro, Parquet, CSV, text, json, ORC to create tables in spark.
		* 
	* [[Structured Streaming]]
	* [[GraphX]]
* Pillars of core philosophy 
	* Speed
		* The queries are built as a [[DAG]] (Directed Acyclic Graph)
		* The [[DAG]] Scheduler and query optimiser create a computational graph. This can be broken into tasks which can be distributed across workers.
		* Generates a compact code for execution.
	* Ease of use
		* Provides low level [[RDD]] (Resilient Distributed Dataset) and high level DataFrames and Datasets.
		* Provides operations in terms of actions and transformations.
	* Modularity
		* Comes from unified APIs for variety of workloads.
	* Extensibility
		* Can read data from wide variety of sources.
		* 



* Layer model for Spark:
	* Physical layer
	* Metadata layer
	* Compute Layer: Spark Cluster
