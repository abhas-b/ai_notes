 #aws [[AWS 1]][[Data Engineering]]

## Governance 

* It covers how data is managed through its lifecycle including security, compliance and effectiveness. It includes processes that make the data discoverable, understandable and usable while ensuring data quality.

## Security 

* Authorization and encryption are key components of security. 
* [[S3]] Policy can be
	* User based
	* Resource based
		* 
	* Both can be combined as well.

## Core concepts

* PII
* Personal data
* Encryption :
	* in transit( includes endpoint authentication alongside encryption). This is done using TLS protocol (transport layer security)
	* Encryption at rest
		* Server side encryption (SSE)
			* AES 256
			* This is done by [[S3]] for authorized users. Also does the decryption.
				* SSE - S3 : S3 manages the key. If the bucket is accidentally made public and it allows anonymous users then SSE-S3 will decrypt the object even for anonymous users.
				* SSE - KMS: [[KMS]] manages the key. S3 can be configured to use it. To use the key the client needs permission for S3 bucket as well as the KMS key. 
				* SSE - C: the client maintains the key. So every request to S3 must include the key. 
	* Another classification
		* Client side encryption
* Anonymized data: masking, removal of partial data.
* Tokenization / pseudonymized data
* Authentication 
* Authorization 
* Least privilege access


## Data quality, profiling and lineage

### Quality 
* This impacts trust and data sharing.
* Measuring quality: accuracy, completeness, consistency, timeliness.
* You can create a DQ check job as part of the data pipeline where you specify the rules for quality checks and generate quality report.

### Profiling 
* Analysing a dataset and reporting on various aspects of the data.
* This is done in terms of metrics on the data such as missing values, distinct values, range of values, max and min of string length
* Profiling is related to quality but more about the shape of the data. This assists various users to assess whether the dataset is suitable for their application or not.

### Data Lineage

* Captures how a dataset has been created including sources and transformations.
* Can be recorded in data catalog. 

## Business and Technical catalogs [[Glue]]
 
 * Cataloging makes the data discoverable.
 * Adding useful metadata is important for building trust in a dataset. 
 * Enforcing rules such as meeting metadata requirements for adding to catalog and data quality standards help in building a good data platform.
 * A business catalog is usually associated with one or more technical catalogs.
### Business Catalogs (data catalog )

* It stores metadata about various datasets and acts as a central repository.
* Assists dataset discoverability.
* Metadata can include: description, tags, owning team, data owner, update frequency, last updated time, data quality score, lineage, profile etc.
* The dataset entry is also linked with a technical catalog which contains details of each field in the dataset.
* DataZone service can be used as an in-house solution.

## Technical Catalog 
* Maps physical files into a logical representation as databases and tables.
* Information stored can be schema, physical location of the files and partition information, file format.
* Glue Data Catalog is a hive compatible metastore.
* Catalog is used by Athena, Glue, EMR.

## AWS services for governance 

### AWS Glue/Lake Formation technical data catalog 
* Maps physical files to logic representation.
* Stores schema and metadata about tables in other databases such as Redshift, RDS, DDB etc.
* Lake formation also provides an interface for the same catalog in addition to glue data catalog.
* Both tools are similar in design and functionality.
* Glue crawlers can be run to examine a data source infer schema and update the catalog.
* Workflow should include crawler to run after new dataset is available or glue api is used for the same.
* Neither of these services work with the business catalog. A separate process is needed to add datasets to business catalog.


## Glue Databrew for profiling datasets

* Number of columns, rows, distinct and unique values, max and min values, max and min string length, null or missing values.
* A Databrew profile job can be used to examine and profile a dataset.

## Glue Data Quality 
* Runs on catalogued datasets in S3.
* Can be called directly from Glue Studio.
* Based on Deequ data quality framework and uses Data Quality Definition Language (DQDL) to define quality rules.
* Both glue data studio and data catalog can be used for building the set of quality rules. Databrew also supports a visual editor for this.

## Key Management Service (KMS) for encryption 

* Helps in creating and managing security keys.
* Services that use KMS:
	* Appflow, Athena, EMR, Kinesis, MSK (managed streaming for kafka), MWAA, redshift, S3, DMS, Lambda, Glue, Databrew.
* All use of AWS KMS keys is logged in CloudTrail.

## Amazon Macie for detecting PII in S3 objects 
* Managed service, ML based. Detects and alerts about PII data.
* Response can be automated.

## Glue Studio Detect PII transform
* This can be done during a transform job.
* You can opt to detect sensitive data operation and configure as per requirement.

## IAM 
Policy types:
* AWS managed policies
* Customer managed policies 
* Inline policies 

Traditionally we used IAM to control access to S3 based data lakes. This became a challenge for large data lakes as the json documents would need to be updated manually as and when new locations are added to the lake. Here Lake Formation is useful.

## Lake Formation for data lake access management [[Lake Formation]]


Enables lake access management for databases and tables without having to manage fine grained management using json files.

Apply broader permissions with IAM and then fine grained permissions with Lake Formation.

IAM can be used to manage access to data catalog objects in glue data catalog (databases and tables) as well as physical files in underlying storage. For each object in catalog the user needs permissions to both catalog object as well as S3 permissions for the files.

With the newer permissions model using Lake Formation, you can use IAM to provide access to Glue catalog objects. You don't need to provide access to specific data lake objects as lake formation provides temporary credentials to the analytic services to access S3 data.

It works with: Athena, Redshift Spectrum, Quicksight, EMR, Glue, Sagemaker Studio.

Now you use IAM to control access to service and catalog objects and lake formation takes care of access to specific databases and tables.

By default every database and table haa a special permission enabled that effectively tells Lake Formation to just use IAM permissions and just ignore any permissions that may have been granted in Lake Formation. This is called Pass Through permission.

To enable Lake Formation permissions we need to remove IAM control from the database and table by removing IAMAllowedPrincipals permission.

Once lake formation permissions are enabled we can grant users access to all catalog objects and use lake formation to limit which db and tables they can access.

When using this, we don't have to grant S3 permissions separately as lake formation grants temporary credentials for accessing S3.

We can use lake formation to grant column level permissions.









