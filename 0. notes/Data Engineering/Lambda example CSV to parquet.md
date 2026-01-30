#aws [[AWS/Lambda/Lambda|Lambda]] [[Data Engineering]]

```

import json
import boto3
import awswrangler as wr
from urllib.parse import unquote_plus

def lambda_handler(event, context):
    # TODO implement
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key')
        key_list = key.split("/")
        #print(bucket)
        #print(key)
        print(f'key list:{key_list}')
        db_name = key_list[len(key_list)-3]
        table_name = key_list[len(key_list)-2]
        
    print(f'Bucket:{bucket}')
    print(f'Key:{key}')
    print(f'DB Name: {db_name}')
    print(f'Table Name: {table_name}')
        
    input_path = f"s3://{bucket}/{key}"
    print(f'Input Path: {input_path}')
        
    output_path = f"s3://abhasbaa-demo-dataeng-clean-zone/{db_name}/{table_name}"
    print(f'Output Path: {output_path}')
        
    ## Configure wrangler to convert the file 
    input_df = wr.s3.read_csv([input_path])
    current_databases = wr.catalog.databases()
        
    if db_name not in current_databases.values():
        print(f'=== Creating Database {db_name} ===')
        wr.catalog.create_database(db_name)
    else:
        print(f'--- {db_name} Already exists ---')
            
            
    result = wr.s3.to_parquet(
        df = input_df,
        path = output_path,
        dataset = True,
        database = db_name,
        table = table_name,
        mode = "append"
        )
    print('RESULT:')
    print(f'{result}')
        
    return result
            
            
  
        
       
        
```