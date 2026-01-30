#aws [[EC2]]


# EC2



* Firewalls are configured using Security Groups.
* Security groups contain only allow rules.
* These rules can reference by IP or by other security group.

They regulate:
* Access to ports
* Authorized IP ranges
* Inbound network 
* Outbound network

## Security Groups

* Can be attached to multiple instances.
* An instance can have multiple security groups.
* A sg is locked down to a region/VPC combination.
* By default all inbound traffic is blocked and all outbound traffic is authorized


Standard Ports:
* 22 SSH on Linux instance
* 21 FTP
* 22 SFTP
* 80 HTTP
* 443 HTTPS
* 3389 (RDP: Remote desktop protocol) : windows instance


# S3

* User based
	* IAM Policies: which API calls should be allowed for a specific user
* Resource Based
	* Bucket Policies : bucket wide rules configurable from S3 - allow cross accounts (most common way to manage security)
	* Object Access Control List (ACL) : finer grain, can be disabled as well
	* Bucket ACL
* For EC2, we can use IAM Roles.


* A principal can access an object if either IAM allows or Resource Based policy allows and there is no explicit DENY on the specified API call.
* Objects can also be encrypted.


## S3 Bucket Policies

* JSON based.
* Can be used to grant public access to a bucket.
* Force objects to be encrypted at upload.
* Grant access to another account.
* We can use policy generator for a quick policy doc.
* 

