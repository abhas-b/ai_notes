#aws [[Relational Databases]]

* Relational Database Service
* SQL
* A managed DB service for DBs that use SQL as a query language
	* PostGres
	* MySQL
	* MariaDB
	* Oracle
	* MS SQL Server
	* IBM DB2
	* [[Aurora]]
* Features
	* Automated provisioning, OS patching
	* Continuous backups and point in time restores
	* Monitoring dashboards
	* Read replicas for improved read performance
	* Multi AZ setup for Disaster Recovery
	*  Storage is backed by EBS
* Deployment Options
	* Read Replicas
		* Allows us to scale read workload of DB
		* Upto 15 replicas
		* Writing is done only to the main DB
	* Multi-AZ
		* Used for DR
		* Allows failover to the replica in another AZ in case something happens to main DB
	* Multi Region
		* Read replicas
		* 
