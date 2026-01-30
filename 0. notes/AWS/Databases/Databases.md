[[AWS 1]] #aws 

* Allow for more efficiency in retrieving data due to indexes compared to EBS, EFS, Instance store, S3.
* You can define relationships between datasets.

1. Relational Databases [[Relational Databases]]
	1. SQL is used here
2. No SQL databases [[DynamoDB]]
	1. built for a specific data models and have flexible schemas for building modern applications.
	2. Examples: key value (JSON), document, graph, in-memory, search dbs.
3. In memory databases: [[ElastiCache]]
4. Redshift: Based on PostGres, but not meant for OLTP (Online Transaction Processing) but for OLAP (Online Analytical Processing).
5. EMR: Elastic Map Reduce
	1. Used to create Hadoop clusters
6. Athena
	1. Use cases: BI, analytics, reporting, Analyze and query VPC flow logs, ELB logs, CloudTrail trails.
	2. Its serverless.
7. DocumentDB
	1. Cloud native version of MongoDB
	2. Basically Aurora version of MongoDB
8. Neptune: fully managed graph database. 3 AZ, 15 read replicas.
9. Timestream: time series. 
10. QLDB: Quantum Ledger Database
	1. Used for financial transactions
	2. Review history of all the changes made to your application data over time
	3. Immutable system: entry cannot be removed/modified. This is cryptographically verifiable.
	4. Supports SQL
	5. There is no concept of decentralization in the ledger.
11. Managed Blockchain: decentralized blockchain.
12. 
