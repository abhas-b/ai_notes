
#aws  [[AWS 1]]

* We can run Docker containers on EC2.
* Amazon ECR (Elastic Container Registry) is a private docker image repo.
	* Can be referred by both ECS and Fargate
* ECS: Elastic Container Service
	* Used to launch Docker containers on AWS
	* You must provision and maintain the infra (EC2 instances) in advance
	* Also supports ALB
* Fargate:
	* Also used to launch Docker containers
	* No need to provision infra
	* Serverless offering
