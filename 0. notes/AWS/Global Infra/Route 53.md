#aws 

* Managed DNS
* Mapping categories
	* hostname to iPv4: A record
	* hostname to iPv6: AAAA record
	* hostname to hostname: CNAME
	* hostname to AWS resource: Alias
* Routing Policies
	* Simple Routing Policy
		* No health checks
	* Weighted Routing Policy
		* Allows us to distribute traffic across multiple instances basis the weights that we assign.
		* We can use health checks here.
	* Latency Routing Policy
		* Useful for global applications.
		* Takes user location into consideration.
	* Failover Routing Policy
		* Used for Disaster Recovery
		* 

