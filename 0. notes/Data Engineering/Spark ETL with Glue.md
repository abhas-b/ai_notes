#aws [[Glue]]  [[Spark]]  [[ETL]]


* Permissions: 
	* AWSGlueServiceRole [[Frequent Permissions]]
	* Glue works with Spark Cluster here. So we'll need to pass on the permissions to the cluster as well.
	* For this: AWSGlueServiceNotebookRoleDefault [[Frequent Permissions]]. This needs iam:PassRole Action. This can be done as inline policy as well. Note that general services take up stsAssumeRole.
	*


[[Spark Commands]]

```
# Create dynamic frame from catalog
dynamic_frame_df = glueContext.create_dynamic_frame.from_catalog (
		database = "",
		table = "",
		transformation_ctx = "custom bookmark name"
)

```


 ```
 # Convert dynamic frame to spark df
 result_df = dynamic_frame_df.toDF()
```


For using SQL interface we need to publish df as a temporary view.

```
result_df.createOrReplaceTempView("table_name_that_you_want")
spark.sql("""select --- from table_name_that_you_want""")
```


Write back the processed df to S3:
```
clean_dynamic_frame = DynamicFrame.fromDF(final_df, glueContext, 'ctx')
glueContext.write_dynamic_frame.from_options
(
frame = clean_dynamic_frame,
connection_type = "s3",
format = "csv",
connection_options = {
"path":"s3 folder parh",
"partitionKeys":[]
},
transformaion_ctx = "xtc"
)
```


f