
[[AWS 1]] #aws 


EC2 provides scalable compute capacity. Basically provides CPU and memory. (Elastic Compute Cloud).

Provides IaaS on AWS.

Capabilities:
1. Virtual Machines (EC2)
2. Virtual storage on drives (EBS) [[Network Storage - EBS and EFS]]
3. Load balancing (ELB) [[ELB]]
4. Auto scaling [[ASG]]

Available Config options:
* OS
	* Windows, Linux, Mac
* Compute Power and cores (CPU)
* RAM
* Storage
	* Network attached ([[Network Storage - EBS and EFS]])
	* Hardware attached ([[EC2 Instance Store]])
* Network card: speed and networking
* Firewall rules (security group)
* Bootstrap script: EC2 User Data
	* Automate boot tasks (such as starting a web server, software/update installation)


## Instance types:

![[Pasted image 20240601130356.png]]

free tier: t2.micro, 750 hours per month

* Key Pair for the instance:
	* Encryption: RSA
	* File format:
		* Mac, Linux, Windows 10: .pem format. This can leverage SSH
		* Otherwise .ppk format (requires PUTTY) (< Windows 10)

* For security and SSH, we need to create a Security Group.
	* a default launch-wizard-1 will be created with standard settings at the time of instance creation.

 * Storage: 
	 * Free tier eligible customers can get up to 30 GB of EBS General Purpose (SSD) or Magnetic storage


If you stop an instance and re-start it, AWS will change its public IP v4. Private IP wont change.

## Instance Types

* General Purpose t series
	* web servers
	* code repositories
* Compute Optimized - c series
	* high performance processors
	* batch processing workloads
	* media transcoding
	* ML
	* gaming
	* HPC
	* high performance web servers
* Memory Optimized r series
	* large datasets in memory
	* databases
	* distributed web scale cache stores
	* in memory db for BI
	* real time processing of unstructured data
* Accelerated Computing
* Storage Optimized i/g/h
	* for tasks requiring high, sequential read and write access to large data sets on local storage
	* OLTP
	* Relational and NoSQL db
	* cache for in-memory db (Redis)
	* DWH
	* Distributed file systems
* Instance Features
* Measuring Instance Performance

 Naming Convention:

* m5.2xlarge
	* m: instance class
	* 5: generation
	* 2xlarge: size within instance class

Review all instances here: [link](https://instances.vantage.sh/)

Instances are bound to an AZ.
