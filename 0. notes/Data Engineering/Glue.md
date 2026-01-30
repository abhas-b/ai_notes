#aws [[AWS 1]][[Data Engineering]] [[ETL]]


DataFrames
* Glue has its own version of dataframes known as Glue Dynamic Frame.
* It is a ETL optimized data structure, similar to Spark Dataframe.
* It integrates with Glue Data Catalog and AWS connectors.
* It can handle a field with mixed data types. This allows Dynamic Frame to handle data inconsistencies.
	* A field with some integer and some string values would have a data type [string, integer]. This is called a choice or union type. Spark would have stored this as string.
	* We can resolve these inconsistencies after dataframe creation to make the data compatible with the destination system.
	* This is done using a resolve choice method.
		* Solutions:
			* We can cast all values to a single data type
			* Create separate columns for each data type.
* We can convert between a DynamicFrame and Spark dataframe (toDF method) and vice versa (from DF method).
	* A convenient use case is using DynamicFrames for connections with integration touchpoints and convert to Spark dataframes for processing.
* There is a parameter for methods called "Transformation Context"
	* If specified - persist state for that operation (bookmark)
	* If not specified, job bookmarks are not enabled for that operation and state is not tracked.




Components:
* Jobs/ETL
	* Job types:
		* Python shell
			* For ETLs that do not need Spark
		* Spark jobs
			* Suitable for batch processing
			* Runs on serverless Spark cluster
		* Streaming ETL
			* Streaming data from Kafka/Kinesis.
			* Load data to data lake or database
		* Ray
			* new framework for ML workflows
	* Properties 
		* Source 
		* Target
		* Transformations
		* Scheduling
* [[Glue Crawlers]]
	* Classifies data based on file format, schema.
	* Groups data into tables and databases.
	* Scan the data sources and update the catalog with schema of the data.
	* Once updated the catalog can be used for queries and ETL.
	* You'll need a database and a table to scan data from source for updating catalog.
	* By default the crawler uses LazySerDe. This cannot handle embedded quotes in the data and can lead to DQ issues. AWS recommends using [[OpenCSVSerDe]] in such cases.
	* In a given folder, files organized across multiple files maybe treated as partitions or individual tables, based on their schema. Basically crawler can create multiple tables from the same S3 prefix. But if they are in same bucket structure then [[Athena]] doesn't like it (nor do [[EMR]], [[Redshift Spectrum]])
	* [[Athena]] expects each table to map to one S3 path that contains files with a similar schema. Hence using individual crawlers for each S3 prefix is a better approach.
* [[Glue Data Catalog]] 
	* Metadata catalog
	* Stores information about data sources and their relationships as well as transformations on data.
	*  Components 
		* Databases
			* They store tables in the catalog.
			* Can contain one or more tables.
		* Tables
			* Stores information about data sources such as schema, data types and relationships between various data elements.
		* Columns
			* These define the structure of the table.
			* A column represents a data element in the source.
		* Partitions 
			* Used to make queries more efficient.
			* When you add or remove partitions then you need to update the catalog.
			* Partition updates
				* Run [[Glue Crawlers]] (ad-hoc or scheduled). Handles both hive compatible partitions as well as regular folder names.
				* [[MSCK]] command in [[Athena]]. This is faster then crawlers but useful only for Hive compatible partitions (folders need to be named as key-value pairs). Can also be executed by running [[Lambda]]
				* ADD, DROP partition commands. Its a low level table command (fastest). Can be used with [[Lambda]]
		* Classifiers
			* Used to identify data format (CSV, avro)
			* Classifiers can automatically extract schema of data and store it in catalog.
		* Connections
			* Defines relationship between data source and catalog.
			* Includes authentication information.
		* 
* Workflows
	* Used for orchestration.
* [[Glue Triggers]]
	* Allow job scheduling 
	* Options:
		* Event driven
		* Scheduled
		* Manual 
* Dev Endpoints
	* It's a dev environment that allows for testing and debugging the ETL jobs before deployment to production.
* [[Job bookmarks]]
	* Persists state between job runs.
	* Allows job restart from the point of failure by tracking data that has been processed before failure.
	* Bookmarking strategy depends on the data source
		* S3: last modified time of the object
		* JDBC sources such as relational database: primary key of the table is used as bookmark key or a specific column.
	* Any column can be used as a Bookmark key. Bookmark key should increase or decrease over time (monotonous) without any time gaps.
	* States
		* Enabled: enables initial full load + subsequent incremental loads
		* Disabled: full load at each run
		* Paused: incremental data since last successful run without updating the state.

Use cases:
- DW migration: from on Prem to S3 or Redshift.
- Data Lake migration.
- Data Transformation 
- Data Enrichment 
- Data Cleaning 
- Data Integration from various sources.


Other services typically used with Glue:
* [[IAM]]
* [[SNS]]
* [[CloudFormation]]
* [[KMS]]
* [[Cloudwatch]]


