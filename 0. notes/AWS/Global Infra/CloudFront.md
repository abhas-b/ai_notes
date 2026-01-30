#aws 

* Its a content delivery network (CDN).
* It caches the content at the edge and hence improves read performance.
* Origins
	* S3 bucket
		* For distributing files and caching them at the edge.
		* Enhanced security with CloudFront Origin Access Control (OAC) and S3 bucket policy.
		* It can also be used to send data (upload to S3). Here it acts as an ingress.
	* Custom Origin (HTTP)
		* ALB
		* EC2
		* S3 website
		* Any HTTP backend
* CloudFront is useful when you have static content that must be available everywhere.
* The files are cached for a TTL.
* 