#spark [[Spark SQL]]

# Managed Tables vs External Tables

* saveAsTable creates a managed table by default
* create table DDL statement also creates managed table by default.

Managed table: 
1. Spark creates managed table at a pre-defined DWH location.
	1. Can be configured using spark.sql.warehouse.dir variable. This can be done before the cluster starts. Once started, this cannot be altered as this is a static configuration.
2. Spark manages metadata and table data together.
3. When you drop the table, both data and metadata are deleted.


External Table:
1. Also known as unmanaged table.
2. Its a spark table created over shared external data.
3. They can be used for sharing data across projects or storage layers.
4. We can use external table to use data stored in some external location with data in current location. This avoids data replication.
5. If you provide location clause, then spark will create an external table. Without location spark creates managed table.
6. Since we create external table at same location as the data, it doesn't need to replicate the data and we dont need to load data into external table after creating it.
7. When you delete the external table, spark only deletes the metadata. It doesn't manage the data.



