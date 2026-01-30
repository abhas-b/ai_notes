#aws [[AWS 1]] [[Serverless Computing]] [[Data Engineering]]


* Event driven serverless Computing platform.
* Code is called Lambda function.
* Runs in containerised environment. They run only when triggered and no resource is consumed otherwise.

* The invoking services require the necessary permission to invoke the function.
* The permission to invoke comes from IAM policy/role.
* 
* The event object holds the input data for the function on. Structure depends on triggering source.
* Invocation types: synchronous asynchronous. This depends on source.
* S3 is always async. API gateway is synchronous.
* We can control invocation type using SDK only. 

Types of lambda event sources:
* Push based events (S3, api gateway )
* Pull/Poll events (ddb stream, kinesis, sqs)