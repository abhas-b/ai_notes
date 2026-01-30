[Link](https://www.youtube.com/watch?v=ZvJSaioPYyo&t=3s)

[[Glue]] [[Data Engineering]]

* AWS Glue is serverless.

* Catalog
	* Persistent Data Metastore.
	* It is a managed data store which can be used to query and transform data.
	* One AWS Data Catalog per AWS region.
	* Can be used for data governance via data annotation.
	* Stores metadata:
		* File Location
		* Schema
		* Datatypes
		* Data Classification
* A database is a set of data catalog table definitions organized into a logical group.
* Glue Tables: metadata definition representing your data. The data stays in its store location. This is just logical representation of the schema.
* Crawlers: A program that connects to a data store (source/target), progresses through a prioritized list of classifiers to determine the schema for your data and then creates metadata tables in the Catalog.
* Partitions: S3 folders (physical entities) are mapped to partitions. Partitions are logical entities in AWS.
	* Partitioning speeds up queries.
* Glue Connections: a data catalog object that contains the properties that are required to connect to a particular data store.


ETL
* 1 DPU = 4 vCPUs and 16 GBB memory.
* 