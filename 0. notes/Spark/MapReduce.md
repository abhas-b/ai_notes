#data-engineering [[Spark]][[Data Engineering]]

* MR is basically functional programming 
* MR sends the code to where the data resides. This brings data locality and cluster rack affinity.
* It writes the intermediate outputs to disk on nodes and aims to reduce network I/O.
* Issues with MR
	* Required a lot of boilerplate code and had limited fault tolerance.
	* Each MR pair's intermediate output had to be written to disk for the subsequent stage. This frequent disk I/O increased computation time.
	* Had limitations wrt modern workloads: streaming, ML.
* 