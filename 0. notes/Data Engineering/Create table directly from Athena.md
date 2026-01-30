#aws [[Athena]]


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
ROW FORMAT SERDE 
  'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' 
STORED AS INPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' 
OUTPUTFORMAT 
  'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
LOCATION
  's3://abhasbaa-demo-datalake-bucket/customer_reviews/parquet/'
TBLPROPERTIES (
  'transient_lastDdlTime'='1676737250')
```


# [[OpenCSVSerde]]

## Version 1: when all values are string only.

```
CREATE EXTERNAL TABLE IF NOT EXISTS `university_ranking_csv`(
  `university` string, 
  `year` int, 
  `rank_display` string, 
  `score` float, 
  `link` string, 
  `country` string, 
  `city` string, 
  `region` string, 
  `logo` string, 
  `type` string, 
  `research_output` string, 
  `student_faculty_ratio` float, 
  `international_students` string, 
  `size` string, 
  `faculty_count` string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ("separatorChar" = ",", "escapeChar" = "\\", "quoteChar"='\"') 
LOCATION
  's3://abhasbaa-demo-datalake-bucket/university-ranking/csv/'
TBLPROPERTIES ("skip.header.line.count"="1")
```


* It can handle missing/null values only for string data.

## Version 2: use string type for all columns then cast to desired data type later. This lets you handle missing data from all columns and then handle exact data type at run time.

```
CREATE EXTERNAL TABLE IF NOT EXISTS `learn_by_doing.university_ranking_csv`(
  `university` string, 
  `year` string, 
  `rank_display` string, 
  `score` string, 
  `link` string, 
  `country` string, 
  `city` string, 
  `region` string, 
  `logo` string, 
  `type` string, 
  `research_output` string, 
  `student_faculty_ratio` string, 
  `international_students` string, 
  `size` string, 
  `faculty_count` string)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
WITH SERDEPROPERTIES ("separatorChar" = ",", "escapeChar" = "\\", "quoteChar"='\"') 
LOCATION
  's3://aws-glue-chandra-us-east-1/university_ranking/csv/'
TBLPROPERTIES ("skip.header.line.count"="1")
```

