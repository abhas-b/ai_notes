#aws [[Databases]]

* Supports replication across 3 AZ.
* NoSQL db
* Distributed serverless database.
* Single digit millisecond latency - low latency retrieval.
* Supports Auto scaling
* Storage classes:
	* Standard
	* Infrequent Access
* Key value database
	* The key contains a partition key and a sortkey
	* The values depends on the entry, since this is JSON like.
* DynamoDB Accelerator - DAC
	* Fully managed in-memory cache for DynamoDB
	* Replacement for ElasticCache for DDB.