#aws [[AWS 1]][[Data Engineering]]

# Transactional data Lakes

### What is a transaction: 
A database transaction represents a unit of work performed within a DBMS against a database that is coherent and reliable way independent of other transactions. This can include multiple updates for a transaction with the guarantee that all individual updates will work and will be applied consistently, or the whole transaction will fail in case of any update fails. That means that if there are five updates as part of the transaction, and the third update fails, then the two previous updates that had been applied will be rolled back, and the last two updates will not be applied. Either everything in the transaction succeeds, or the database is returned to the state that it was in prior to the transaction being attempted.

### Limitations of Hive based Data Lakes:

* Hive could not do multiple updates per transaction.
* For updating a row we had to read the relevant partition, update the row and then rewrite the table or partition.
* For large datasets this needs to read large number of files and using up a lot of compute. 
* Concurrency management during these rewrites was also a challenge. If the data was queried during the update then incorrect data would be returned. Even worse, if two jobs both attempt to update the table at the same time, there is a chance that both jobs fail at some point, and the underlying table, or partition, is left in an inconsistent state.

### Benefits of Open Table Formats 


1. [[ACID Transactions]]
2. [[Time Travel]]
3. [[Record level updates]]
4. [[Schema evolution]]


### How OTFs work

In formats like parquet the metadata is stored alongwith the actual data. in the file, along with the actual data. This includes metadata about the file (such as number of columns and rows in the file) and column metadata (such as the data type stored in the column, as well as statistics such as the min and max values for that column). This facilitates query optimization.

The OTFs also store metadata about the files and tables. Each format tracks the current state of the table as well as a history of how the table has evolved. Some key stats are also stored such as column level metrics and descriptive stats. These stats enable the query planner to identify file paths with relevant data without opening the file.

### Approaches used by OTFs for updating tables

1. COW : copy of write
	1. Whenever a record is updated or deleted a new copy of the files are created with updated data.
	2. Metadata is updated to reflect that there is a new version or snapshot of the data.
	3. This metadata versioning enables time travel as each version points to source files corresponding to that version only.
	4. Since each files needs to be recreated, cow has a read performance impact.
	5. COW is write optimized and suitable for cases when updates/deletes are infrequent.
2. MOR : merge on read
	1. Files are not rewritten. Instead if a record is deleted or updated then a deleted tracking metadata file is created to cr










