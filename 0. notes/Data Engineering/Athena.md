#aws  [[Data Engineering]]


* Pricing is based on data scanned or provisioned (based on capacity reservation).
* Also supports PySpark based notebooks.
* There needs to be a table: either through crawler or Athena DDL.
* Amazon Athena can also transform files using a concept called Create Table As Select (CTAS). With this approach, you run a CTAS statement using Athena, and this instructs Athena to create a new table based on a SQL select statement against a different table.
*```
 CREATE TABLE customers_parquet WITH (  format = 'Parquet',  parquet_compression = 'SNAPPY') AS SELECT * FROM customers_csv;

* It is important to avoid having a large number of small files if you want to optimize your analytic queries.
* While there are no hard rules regarding file size, it is generally recommended that to optimize for analytics you should aim for file sizes of at least 128 MB for objects in your S3-based data lake.
* When using file formats such as Parquet that are splitable (meaning the query engine can split the file and process different parts of the file in parallel), the maximum file size is not as much of an issue. But if processing files that are not splitable, such as Gzipped CSV files, it is important to have multiple files that the query engine can read in parallel, rather than a single large file.
* Bucketing: Bucketing is a concept that is related to partitioning. However, with bucketing, you group rows of data together in a specified number of buckets, based on the hash value of a column (or columns) that you specify.
### Connectors 
* Federated queries use lambda with connectors to connect to various data sources. 
* When a query runs that uses a connected data source, Athena invokes the relevant Lambda function/s to read metadata, identifies parts of the tables that need to be read, and launches multiple Lambda functions to read the data in parallel.
* A JDBC connector for connecting to sources such as MySQL, Postgres, and Redshift. A DynamoDB connector for reading from the Amazon-managed NoSQL database. A Redis connector for reading data from Redis instances. A CloudWatch logs and CloudWatch metrics connector, enabling you to query your application log files and metrics using SQL. An AWS CMDB connector that integrates with several AWS services to enable SQL queries against your AWS resources. Integrated services include EC2, RDS, EMR, and S3. A Google BigQuery connector that enables queries across clouds for data in Google BigQuery tables. An Apache Kafka connector that enables querying real-time data in Apache Kafka (and Amazon MSK) topics. With this connector you can join data in a Kafka topic with data in other topics, or with data in your Amazon S3 based data lake.

### Capacity Provisioning 
- When using provisioned capacity, you are charged based on the compute resources you provision, and there is not a per query charge based on data scanned.
- When you provision capacity for Athena, you get dedicated processing capacity for your queries. To use this functionality, you specify the number of Data Processing Units (DPUs) that you need for your queries, and you assign one or more Workgroups to use that capacity. One DPU provides for 4 CPUs, and 16 GB of memory.
- For 10 concurrent queries, Amazon recommends 40 DPUs For 20 concurrent queries, Amazon recommends 96 DPUs For 30 or more concurrent queries, Amazon recommends 240 DPUs.


[[Create table directly from Athena]]
* We can create table either through crawler or directly via Athena.
* Sample command for the same:

```
CREATE EXTERNAL TABLE `amazon_reviews_parquet`(
  `marketplace` string, 
  `customer_id` string, 
  `review_id` string, 
  `product_id` string, 
  `product_parent` string, 
  `product_title` string, 
  `product_category` string,  
  `star_rating` int, 
  `helpful_votes` int, 
  `total_votes` int, 
  `vine` string, 
  `verified_purchase` string, 
  `review_headline` string, 
  `review_body` string, 
  `review_date` string,   
  `sentiment` string)
ROW FORMAT SERDE -- for parquet files
  'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' 
STORED AS INPUTFORMAT  -- for parquet files
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' 
OUTPUTFORMAT  -- for parquet files
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
LOCATION -- update
  's3://your-bucket/customer_review_parquet/'
TBLPROPERTIES (
  'transient_lastDdlTime'='1676737250')
```

[Reference link](https://github.com/ChandraLingam/DataLake/tree/main/CustomerReview/scripts)


* You should keep similar files in a particular S3 path. Don't mix and match between different structures.



[[MSCK]]:

```
 MSCK REPAIR TABLE "demo_db.iris_partitioned"
```

* MSCK does not update catalog. For updating catalog run following:

```
ALTER TABLE iris_partitioned ADD PARTITION (partition_0:'versicolor') location 's3:...';

-- Here we are adding versicolor partition to existing table.
```
