#aws  [[EC2]]


# AMI: Amazon Machine Image.

* Its basically a bundled customized configuration for an instance. Like OS, tools etc.
* Speeds up the boot time since all software is pre-packaged and doesn't need to be installed at boot time.
* AMI are built for a specific region and can be copied across regions.
*  Sources of AMIs
	* Public AMI 
	* Custom AMI
	* AMI marketplace


# EC2 Image Builder

* Used to automate the creation of VMs.
* This can help automate the creation, maintenance and validation of AMIs.
* When it runs, it creates a Builder EC2 Instance which builds components (installs software, firewalls etc on instance). Then new AMI is created using this Builder. It then creates a test EC2 instance on this AMI. Test cases can be defined in advance.
* It can be run on a schedule.
* Its a free service.


  

