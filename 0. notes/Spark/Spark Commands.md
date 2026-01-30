#spark [[Spark]]

```
from pyspark.sql import SparkSession  
  
if __name__ == '__main__':  
    spark = SparkSession.builder\  
                .appName("Hello Spark") \  
                .master("local[2]")\  
                .getOrCreate()  
  
    data_list = [("ravi", 32)]  
  
    df = spark.createDataFrame(data_list).toDF("Name", "Age")  
    df.show()
```


### Create DataFrame [[2. Spark Data Frames and Tables]] [[Ingestion]]
```
fire_df = spark.read \  
            .format("csv") \  
            .option("header", "true") \  
            .option("inferschema", "true") \  
            .load(r"C:\Users\texter\Desktop\Desktop 2\work_\CodeRepo\DataSets\Fire_Department_Calls_for_Service.csv")
```

The read here is an attribute of the spark session. This attribute gives us a DataFrameReader object.


### Convert DataFrame into temporary table view [[Spark SQL]]
```
fire_df.createGlobalTempView("fire_service_calls_view")
```

Then use as following:
```
select * from global_temp.fire_service_calls_view limit 10
```

