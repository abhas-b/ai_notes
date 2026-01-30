#aws #data-engineering [[Glue]]


Typical services used with Glue:
1. [[S3]]
2. [[KMS]]
3. [[IAM]]
4. [[Cloudwatch]]
5. [[CloudFormation]]
6. [[Redshift]]
7. [[SNS]]

# 1. IAM
* Setup user group (admin access)
* Setup user
* Add relevant role.
	* Add policy such as AWSGlueServiceRole
	* Create a custom policy for granting access to S3 bucket.

# 2. KMS
* Used for SSE.

# 3. SNS
* Create topic
* Create subscription to the topic.
* Confirm subscription

# 4. S3
* As bucket policy: we denied bucket access to all principals, but added a condition to allow our designated role:

```
"Statement": [
{
"Sid":,
"Effect":,
"Principal",
"Resouce":[],
"Action",
"Condition":{
			"ForAnyValue:StringNotLike": {"aws:PrincipleArn":"Role ARN"}
			}
}
]
```

# 5. CloudFormation

* Any resource which can be created using console can be created using CF.


# 6. Crawlers
* Use Classifiers
* Update partition and schema information in the catalog.


# 7. Glue Job
* Usually we have a S3 bucket which stores ETL script.
* This is also called ETL artifact.
* Parameters:
	* DPUs
	* Python version
	* 