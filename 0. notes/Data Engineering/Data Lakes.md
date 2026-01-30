#data-engineering [[Data Engineering]]

Various factors to consider in a data lake
* Cost: storage, processing and querying
	* [[Cost Optimization]]
* Data Governance
* Data cataloging
* Sources
* Quality assessment
* Data Transformation
* Integration with other tools
* Some common data formats:
	* Row format
		* csv, tsv, json, avro
	* Column
		* parquet, orc
* Querying the data
	* In-place Query
		* [[Athena]]
		* [[Redshift Spectrum]]
	* Loading into DW and then query
		* [[Redshift]]
	* Streaming data [[Structured Streaming]]
		* [[Kinesis]]



* OLTP: Online Transaction Processing
	* a type of data processing which executes a number of concurrently occurring transactions.
	* examples: e-com, payments, banking, messaging, video, file downloads, social media
	* It is optimized for real time updates.
	* Involves inserting, updating, deleting small amounts of data in a data store to collect, manage and secure the transactions.
	* Make intensive use of indexing for faster access/response.
* OLAP: Online Analytical processing
	* involves querying the transactions / records in a database for analytical purposes. (insights and reporting).
	* Workloads are usually read intensive.

* OLTP is used to create pipelines to bring data into DW.

Data Problems:
* Data Collection and Ingestion
* Data Storage and Management
* Data Processing and Transformation
* Data Access and Retrieval
* Security and Access control
* Scheduling and Workflow management
* Data catalog and metadata
* Data lifecycle and governance
* Operations and Monitoring


3 Vs of Big Data
* Variety
* Velocity
* Volume

