#data-engineering [[Data Engineering]]

* Simplifies the process of building and running applications inside containers.
* Containers are lightweight, self sufficient environments and package an application along with its dependencies, libraries and configurations.
* Containerization is a lightweight form of virtualization.
* Main concepts in Docker:
	* Dockerfiles
		* Text file that contains all commands, in order(sequence) needed to build a given image.
		* Used by Docker Build command to create a Docker image.
		* 
	* Images
		* These are read only and immutable and contain the configuration and other details required for the application.
	* Containers
		* It is a runtime instance of a Docker Image.
		* It is completely isolated from the host and other containers.
		* It has its own filesystem and can be started, stopped and deleted independently.
* A container exists only as long as the process inside it is alive. Post that the container exits.


# Commands

* docker run nginx
* docker ps -- list all running containers
* docker ps -a -- list all containers
* docker stop <container id / name >
* docker rm -- remove container
* docker images -- list available images on host, with sizes
* docker rmi < image name > -- remove image from host
* You must stop and delete all containers before deleting an image
* docker pull < image name > -- pull image, dont run container
* docker run ubuntu sleep 5: executing a command on running a container
* for running  command from within a container:
	* docker exec < container name > < command >
* docker run -t < image > < command > -- move inside container
* docker run -d .... -- run container in background
* 