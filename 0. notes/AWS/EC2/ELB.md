[[EC2]] #aws 

Scalability:
1. Vertical
	1. increasing the size of the instance
	2. common in non distributed systems such as databases
2. Horizontal (elasticity)
	1. Increase/Decrease the number of instances
	2. Impacts availability as well
	3. Also implicit is auto scaling based on load. This is pay per use.

High Availability:
* Means running your application in at least 2 AZ.


Agility:  depends on time to resource availability after demand is placed.

# Load Balancing

* Allows us to expose a single DNS to the application and at backend multiple servers can attend to the traffic.
* Builds redundancy.
* Can be used across multiple AZ. This improves availability.

# ELB : Elastic Load Balancer

* Its a managed service. So you don't need to provision servers. AWS takes care of the provisioning.

Types of Load Balancers:
* Application Load Balancer (ALB)
	* For HTTP/HTTPS/gRPC only (Layer 7)
	* Supports static DNS (URL) and HTTP Routing
* Network Load Balancer (NLB)
	* Ultra high performance
	* Allows TCP and UDP (Layer 4)
	* Gives Static IP through Elastic IP
* Gateway Load Balancer (GWLB)
	* Layer 3
	* Geneve Protocol on IP Packets
	* Route traffic to firewalls on EC2
	* Intrusion detection
	* It actually doesn't do load balancing for application. It does it for 3rd Party security applications that stand in the way of the gateway and application.
* Classic Load Balancer (Retired now)


 
