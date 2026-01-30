
#aws [[Deployment]]

* Helps manage EC2 and on-premises systems.
* Since it works with both AWS and on-premises systems, it is a hybrid service.
* Features:
	* Supports patching automation.
	* Run commands across entire fleet of servers.
	* Stores parameters configuration with SSM Parameter Store.
* Works via SSM Agent installed on the systems.
* Installed by default on Amazon Linux AMI

## SSM Session Manager

* Allows us to start a secure shell on EC2 instances and on-premises servers. No SSH is needed. This provides better security.
* IAM role: AmazonSSMManagedInstanceCore


## SSM Parameter Store

* It provides secured storage for configuration and secrets.
* It can store API keys, passwords, configurations etc.
* Serverless.
* Controls access permissions using IAM.
* Provides version tracking and optional encryption.
* 


