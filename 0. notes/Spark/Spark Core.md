#spark 

* Consists of 
	* Spark Distributed Compute Engine 
	* Core APIs.

* Spark does not manage the cluster. For this we need a cluster manager.
	* Most common is Hadoop YARN
	* Mesos
	* Spark Standalone Cluster Manager
	* Kubernetes

* Spark also doesn't have distributed storage
	* HDFS
	* S3
	* Blob
	* Cassandra

Spark handles:
* Breaking tasks into smaller tasks
* Scheduling
* Parallel processing
* Fault tolerance upon failure


Consists of 
* [[SparkSession]]

```
from pyspark.sql import SparkSession
```

	* SparkSession is the entry point of working with Spark SQL.


## Spark vs Hadoop

[[Hadoop Fundamentals]]


![[Pasted image 20250813090140.png]]

# RDD

[[RDD]]

* RDD represents raw data.
* Dataframe and Dataset add schema to the raw data.
* The SQL, DataFrame and Dataset operations are optimized by Catalyst and converted to low level RDD operations.



![[Pasted image 20250813090245.png]]

![[Pasted image 20250815161827.png]]

## DataFrames, RDD and Distributed Partitions

* Both SQL code and DF code is passed to Spark SQL Engine.
* From there its passed to Catalyst optimizer.
* Catalyst creates an execution plan (RDD DAG)
	* The DAG is basically layout of step by step approach to how Spark will execute the code using RDD API.
* Internally both Spark Tables and DF are same as RDD.
	* Both DF and table are a logical view of an RDD.
* RDD itself is made up of RDD partitions.
	* The partitions are distributed.

![[Pasted image 20250813091809.png]]

* If one partition is lost then main logical RDD knows how to re-create that partition. Hence the term Resilient in the name.

# Transformations and Actions

* DataFrame and RDD are immutable data structures.
![[Pasted image 20250815164104.png]]

* Transformations: both input and output are dataframes.
* Chain of transformations lead to DAG.
* Transformations are lazy and are not executed until they are terminated by actions.
* Together transformations and actions result in an execution plan.

## Narrow and Wide Transformations
* Narrow dependency transformation: does not have any cross partition dependency.
	* Eg: where, select
* Narrow dependency transformations do not alter the partition structure of the RDD.
* Wide dependency transformations require data shuffle and are slower.


# Spark Jobs - Stages, Task, Shuffle, Slots

```
spark.conf.set("spark.sql.adaptive.enabled", "false") -- disable adaptive execution plans

spark.conf.set("spark.sql.shuffle.partitions", 8) -- Default: 200
```

* Group by results in shuffle (as it is a wide transformation)
* For optimization, number of shuffle partitions should speak to grouping key.


Execution Plan :

Job > Stages > Task


* Spark can construct dependency trees for each action.
* Each action and its graph can be assigned a job to run.


![[Pasted image 20250818090013.png]]


* For each job, spark first creates a logical plan.

![[Pasted image 20250818090036.png]]

* Then this plan is broken into stages.
* Stages are broken at wide dependencies.
* For n wide dependencies, you will have n+1 stages.
* The last operation for each stage would either be a wide transformation or an action.




![[Pasted image 20250818090237.png]]

* Stages are not run in parallel, since they are sequential in nature (output of Stage 0 is input for Stage 1).
* For parallelism, each partition is assigned to a Task.
* Output of each stage is stored in an Exchange buffer. (write exchange)
* Next stage also reads from its exchange (called read exchange).
* Write exchange and read exchange could be on different worker nodes.
* There needs to be data replication between the two exchanges for Stage 1 to be able to read. This replication is known as Shuffle.
* This is where shuffle.partitions from before comes into picture. It decides the number of partitions after the Shuffle operation.
* After this also Tasks are performed on the shuffle partitions.

![[Pasted image 20250818091235.png]]

* Task is the smallest unit of work in Spark Job.
* The tasks are assigned to executors.


# Spark SQL Engine and Query Planning

* One DataFrrame action is assigned one job. Same is done for a SQL expression.
* Once logical plan is compiled in terms of jobs, it is then passed to SQL Engine.

![[Pasted image 20250818092037.png]]


SQL Engine Operations:
1. Analysis
	1. Converts unresolved logical plan into logical plan.
	2. Parses the code for errors and schema using catalog.
	3. If the column names do not resolve or there is a data type mismatch or invalid function name then you might see AnalysisException.
	4. Output: Logical Plan (resolved).
2. Logical Optimization
	1. Applies standard rule based optimizations to the Logical Plan.
		1. Constant folding
		2. Predicate pushdown
		3. Partition pruning
		4. Null propagation
		5. Boolean expression simplification
	2. Output: Optimized Logical Plan.
3. Physical Plan
	1. Creates multiple physical plans.
	2. Then applies Cost model and finds out the cheapest plan in terms of Cost.
	3. For eg: it can use different join algorithms in the different physical plans.
4. CodeGen
	1. Takes the best physical plan and generates RDD code for it.


![[Pasted image 20250818092927.png]]



# How to package and run a Spark application

* Spark application runs on a Spark cluster.
* Spark Cluster types:
	* Hadoop
		* YARN: compute
			* RM and NM

![[Pasted image 20250819085224.png]]

		* HDFS: storage
			* NN and DN
		* Metadata
			* In Memory (catalog)
				* Spark stores DataFrame metadata in an in-memory catalog.
			* Database (meta store)
				* Hadoop installs a MySQL DB on the cluster to store table metadata in Hive meta data store.
				* This is used with a Hive Catalog API.
		* Issue with Hadoop cluster: while YARN and HDFS represent two logical layers in the architecture, they exist on the same physical cluster. So if you remove some nodes, both Storage and Compute capacity take a hit. Here both storage and compute are coupled. To increase 1, we need to increase both.

![[Pasted image 20250819085552.png]]


![[Pasted image 20250819090802.png]]

* In terms of cluster capacity:
	* Master node is not used for compute.
	* Due to DN and NM running on each worker node, we must deduct atleast 1 CPU core and 1 GB RAM from node capacity. 
	* Storage is not impacted by these services. But 5-10% of storage is also used by Disk Buffer and Intermediate Storage. Hadoop maintains 3 copies of data for resilience, effectively reducing available storage capacity to 1/3.


![[Pasted image 20250819091832.png]]


	
	* Non Hadoop
		* De-couples storage and compute.
		* Variants:
			* Spark on single node machine
				* Databricks
				* Local
				* Suitable for development/POC and unit testing.
			* Spark on non-Hadoop Cloud cluster
				* Databricks: compute layer
				* S3: storage layer
		* Spark Cluster Resource Manager
			* YARN
				* Works in Spark Hadoop cluster
			* Kubernetes
				* Containerized Cluster Management Service
				* Control Plane: Master
					* resource controllers
					* scheduler
					* Separate Master Node is not required. Control Plane does that job.
				* Nodes Plane: Workers
					* Workers are contained inside Kubernetes Pods.
				* Available on Google Cloud.
			* Mesos
			* Spark Standalone
				* Used by Azure
				* N Worker nodes
					* Worker service is installed on each worker node
					* This service has the same capability as YARN NM.
				* 1 Master Node
					* Spark Master service is installed here
					* This service has the same capability as YARN RM.
			* All the nodes are VMs.
		* Spark Meta Store:
			* Spark built in metadata
				* Persistent
					* Spark sets up a Workspace Persistent Metdata RDBMS for storing this.
					* This store is persistent implying that it retains metadata even if you shut down the cluster.
					* Its common across multiple clusters. So you can delete your cluster and start new one and still use the same store.
					* 
				* Volatile
					* Derby DB is used to create this.
					* Fast, in-memory.
			* External Metadata
				* Useful for multiple project scenario.



![[Pasted image 20250819093316.png]]

![[Pasted image 20250820093651.png]]


# Running Application Spark Cluster

## Approaches to using Spark

* Interactive approach - typically development
	* Databricks Notebook (Databricks Cloud)
	* Apache Zeppelin (Cloudera Hadoop)
	* Other jupyter nbs (AWS EMR, GCP DataProc)
	* Python IDE - Local Development
	* Pyspark Shell
	* Spark SQL shell

* Spark Job approach - production
	* Spark Job Submit API
		* Typically not used in production directly. Other tools can use it internally.
	* Databricks Notebook Jobs
		* We can submit notebook as an application
	* Workflow management tools

![[Pasted image 20250824143444.png]]

		* Here SP1, 2, and 3 are spark applications.
		* Airflow and ADF are generally used for orchestration of complex workflows.
	* Spark submit command line tool
		* Generally used on Cloudera Cluster, and Databricks for single spark application.
		* It does not create complex workflows.


# Spark Submit Command Line Tool

* Typically a large application would have the following structure:
	* Entry point
		* Python file with main method
		* Can be uploaded to cluster as is.
	* Libraries Folder
		* Developed by dev team
		* Needs to be zipped before upload.
	* Config files
		* to configure and customize the application
		* Can be uploaded to cluster as is.


Example of spark-submit

```
spark-submit --master yarn --deploy-mode client main.py /user/data/sample.txt
```

For large projects:

```
spark-submit --master yarn --deploy-mode cluster --py-files lib.zip --files log4j.properies,spark.conf --conf 'spark.driver.extraJavaOptions=Dlog4j.configuration=log4j.properties' main.py /user/data/sample.csv
```


Help:

```
spark-submit --help
```


# Spark Deployment Modes

* Once spark submit happens, RM coordinates with one NM and starts Application Master container on it.
* The AM container starts application's Main method.
* The main method then creates SparkSession object which then triggers Spark framework.
* The AM then asks RM for more resources for parallel execution.
* Once allocated, AM runs the application in parallel on multiple executors.
* This AM is also known as Spark Driver.


## spark-submit --deploy-mode

* Deploy mode determines the location of the spark driver.
* Client (Deafault)
	* In client mode, driver is a local JVM process.
	* Used by interactive tools since they need to show output on system.
* Cluster
	* In cluster mode, driver is in AM in one of the cluster machines.
	* Recommended for submitted applications.



![[Pasted image 20250824173921.png]]




# Resource allocation to Spark Application

* Options:
```
--driver-cores NUM : Default 1
--driver-memory MEM: Default 1 GB (we typically use 2-4)
--num-executors NUM: Default 2
--executor-cores NUM: Default 1 in YARN and Kubernetes modes. All available cores in standalone mode. Best practice: 4-5.
--executor-memory MEM: Default 1GB
```


![[Pasted image 20250824174522.png]]



# Spark Cluster and Runtime Architecture


* spark-submit is used to submit application to the cluster.
* Spark Programming Languages:
	* Python Language: Pyspark
	* JVM Language: Scala/Java
* Spark is written in Scala which is a JVM language.
* There is a Java wrapper on top of Spark Core.
* There is also a Python wrapper on top of Java wrapper. This is known as PySpark.
* So our Python main method starts a JVM application. Once the application is up and running, the Python wrapper calls the Java wrapper using a Py4J connection.
* Here the PySpark main() is called PySpark Driver. The JVM application is called Application Driver.
* If your code is in Scala, then you wont have the PySpark Driver. But you will always have an Application Driver.
* Each Spark executor is also a JVM application.
* Definition: Spark executor is a JVM process that runs in a resource container on a worker node.


![[Pasted image 20250825090237.png]]


* Executor can work in parallel threads with 1 CPU core per thread.
* These threads are called Slots.


![[Pasted image 20250825090757.png]]


* The driver also has information about the slots.
* Each slot can process 1 data partition at a time via tasks. Each slot basically performs 1 task at a time.
* Task is the smallest unit of work. Slot is the smallest unit of execution.
* The number of tasks at each stage depends on the number of input partitions.
* A task is basically a set of transformations that are being applied to a data partition.
* Hence each task will have 3 components:
	* Input partition
	* Set of transformations
	* Output Partition
		* The slot can send the output partition to the shuffle exchange
			* This happens when stage ends with a wide dependency transformation
		* It can write the output partition to the storage directory
			* Last command is a write command
			* written as a part file
		* It can send output partition to the driver.
			* Last command is a collect command.
			* Doing this with a lot of data which cannot fit in driver's memory can lead to driver crashing with Out of Memory Exception.


Scala Architecture:

![[Pasted image 20250825093108.png]]


PySpark DF API architecture:

![[Pasted image 20250825093143.png]]


PySpark architecture with additional Python libraries (including UDF):

![[Pasted image 20250825093222.png]]

* Here the Python worker is a Python runtime environment used for Python specific libraries.


# Spark Joins

## Internals

```
spark.conf.set("spark.sql.adaptive.enabled", "false") -- disables AWE
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1") -- disables BroadCast Join
spark.conf.set("spark.sql.shuffle.partitions", "10") -- depends on unique join key values
```


* Each dataframe is read into lets say 8 partitions (decided by the driver, with a view of minimizing skew).
* then each df is prepared for processing. The read df is sent to write Exchange.
* The Exchange data will be sorted by the join key and stored as partitions (number of partitions is defined using shuffle.partitions). Hence its best to have number of shuffle partitions as per number of unique join keys.
* The shuffle partitions are saved on the disk at a location internally defined by spark.
* This is done for both right and left dfs for the join. Each reading from file and writing to Exchange is done by individual Stages (Stage 0 and Stage 1 respectively).
* Stage 3 then brings all sorted data into Read Exchange.
* The data movement from Write exchange (shuffle partitions) to read exchange is called shuffle.
* Read exchange is an in-memory exchange. if you need more than available memory, then some data is spilled over to disk and this is managed by Spark internally. In these cases Spark will not throw an out of memory exception.
* We may still get an Out of memory exception:
	* Join stage will read partitions over the network.  Spark reads one partition at a time and buffers it in Network Buffer Memory. If this memory overflows then we can still get an Out of memory exception.
* Then Spark chooses join strategy.
* Number of shuffle partitions is a [[Spark Performance Tuning Strategy]]
* For the read stages you can inspect the UI and check for data skew as part of [[Spark Performance Tuning Strategy]]
* 
