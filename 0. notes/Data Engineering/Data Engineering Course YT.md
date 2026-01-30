[Link](https://www.youtube.com/watch?v=PHsC_t0j1dU&list=PLsGVWPpCYZVmnc9Thf_2OAdHZ82oaVm0z&index=34&t=1772s)

# 1. Introduction to Docker

[[Docker]]

* Simplifies the process of building, shipping and running applications inside containers.
* Used for testing as well as deployments.
* Containerization is a lightweight form of virtualization.
* Main concepts:
	* Dockerfiles
		* a text file that contains all the commands (in sequence) needed to build a given image.
		* Used by docker build command to create a Docker image
	* Images
		* Created from dockerfile
		* These are read only and immutable
		* It is basically a standalone executable package that includes everything needed to run a software, including code, runtime, libraries, environment variables and config files.
	* Containers
		* It is a runtime instance of a docker image.
		* Completely isolated from the host and other containers. 
		* Has its own filesystem.
* 