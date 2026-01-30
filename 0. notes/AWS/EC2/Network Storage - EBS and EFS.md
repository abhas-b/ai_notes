[[EC2]] #aws 

# EBS

Elastic Block Store is a network drive that you can attach to your instances while they run.

* They allow data persistence even after termination.
* 1 volume is bound to a specific AZ. It can be mounted to multiple instances (multi attach).
* Its basically a network USB.
* 1 volume can be detached from 1 instance and attached to another instance.
* To move across AZ, we need to first snapshot it.
* Billing is based on provisioned capacity (size in GB and i/o operations per second).
* EBS doesn't necessarily have to be attached to an instance at creation time. It can be attached on demand as well.

## Snapshots:

* A snapshot is basically a backup.
* You can take the shot even as volume is attached to an instance. But it is recommended to detach it.
* Snapshot can be used to restore data or replicate across AZ/Region
 

### Snapshot Features:

1. EBS Snapshot Archive
	1. Move a ss to an archive tier storage that is 75% cheaper. Takes 24-72 hours to restore.
2. Recycle Bin for EBS Snaphots
	1. Setup rules to retain deleted ss for recovery. (retention: 1 day to 1 year)
3. When we create a volume from snapshot, we can do so in any AZ within a region, but can't move to a different region.





# EFS

* Its a Shared NFS (Network File System) and can be mounted to 100s of EC2 at a time.
* Works only with Linux EC2 instances in multi AZ.
*  Its highly scalable but expensive (3 x gp2), pay per use and no capacity planning.


## EFS Infrequent Access (EFS-IA)

* A storage class that is cost optimized for files not accessed every day.
* Up to 92% cost savings vs EFS-Standard.
* EFS can automatically move data from standard to IA based on lifecycle policy (based on the last time they were accessed).
*  
