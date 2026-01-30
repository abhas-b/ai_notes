#spark [[Spark]]

Components of Hadoop
1. YARN: Cluster Resource Manager
	1. Yet Another Resource Manager
	2. lets multiple programs to be in memory and be executed simultaneously.
	3. Manages the shared resources (CPU, memory, disk i/o)
	4. One node acts as a Master Node and rest of the nodes act as Worker Nodes.
	6. Components
		1. RM: resource manager
			1. Installed on Master Node.
			2. For running the application on Hadoop you must submit the application to the RM.
			3. Once you submit the request the RM asks one of the NMs to start a resource container and run an AM (application master) in the container. The AM container runs your application code.
		2. NM: node manager
			1. Installed on workers
			2. regularly sends the node status report to RM.
		3. AM: application master
2. HDFS: distributed storage
	1. Components
		1. NN: Named Node
			1. Installed on master node.
			2. Sends out a target file to data nodes in parts known as blocks. The typical block size is 128 MB.
			3. Named Node keeps metadata for the blocks. 
				1. File name
				2. Directory location
				3. File size
				4. File blocks
				5. Block ID
				6. Block sequence
				7. Block location
		2. DN: Data Node
			1. Each worker node runs a data node service.
3. Map/Reduce: Distributed Computing
	1. The Hive SQL engine translates all SQL queries into M/R programs.
	2. The resource allocation is taken care of by YARN.
	3. HDFS takes care of data block management.
	4. Programming Model
		1. The desired function is applied to individual file blocks (map).
		2. The individual results are aggregated to give the final result (reduce).
	5. Programming Framework
		1. The framework is a set of APIs that allow you to apply the M/R programming model.
4. Issues with Hadoop:
	1. Everything was based on map reduce.
	2. Intermediate results were written to disk. Due to high Disk I/O, performance was slow.
	3. Spark stores intermediate results in memory, as long as enough memory is available.
	4. For Fault tolerance: Spark maintains lineage which can be used to recompute results in case of failures.

