#data-engineering [[Data Engineering]]

* The ability to change the schema of the table without breaking the ability to query the table.
* This includes modifying column list, changing column name, order of columns, column data type. 
* [[Athena]]: csv/tsv column index based access. Parquet/ORC: column name based access (index based access can also be configured).
* To make Parquet read by index (which lets you rename columns) you must create a table with parquet.column.index.access SerDe property set to true. This forces parquet to behave like csv in terms of schema handling.
* 


