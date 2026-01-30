[[AWS 1]] #aws 

## Overview

* IAM : Identity and Access Management Service
* Groups can only contain users and not other groups.
* Some users can not reside in any group. We can assign inline policies to them.
* Policies are JSON documents. They are used to assign permissions to users/groups/services.
* Apply a principle of least privelege.

Rather than using root account, create a new user and give admin access to it.

## Groups

1. Group name:
	1. admingroup
	2. permissions: 
		1. AdministratorAccess


## IAM Policies:

Inline policy is one that's attached without a group.

Structure:
* Version
* ID: identifier
* Statement
	* Sid: Statement ID (Optional)
	* Effect: Allow/Deny
	* Principal: account/user/role to which this policy applies to
	* Action: list of API calls
	* Resource: could be for example bucket
	* Condition: optional

Account Security:
* Password Policy
* MFA
	* Virtual device: 
		* Authy: multiple tokens on a single device
		* Google authenticator: one phone at a time
	* Physical device:
		* Universal 2nd Factor U2F security Key 
			* Yubi key
				* supports multiple root and IAM users using a single key
		* Hardware key fob MFA device (eg: Gemalto)


## Accessing AWS:
* Web Console  : protected by password + MFA
* CLI : protected by Access Keys (generated through console)
* SDK : access keys
	* Language specific APIs
	* Embedded within application

CLI:
* Access Key ID ~~ user name
* Secret Access Key ~~ password

[[AWS CLI Commands]]

## IAM Roles for Services

Some services (such as EC2, Lambda, CloudFormation) need to perform actions on your behalf. For these actions, they need permissions to the resources. We assign them these permissions through IAM roles.

A role is basically an identity that has specific permissions. They can be assumed by users, applications and services.

When we assume a role STS (security token service) generates short term credentials for us using sts:AssumeRole API call.

## IAM Security Tools:

* IAM Credentials Report: (account level)
	* all users and credentials status
* IAM Access Advisor
	* user level: shows service permissions and their last access timeline.


## Shared Responsibility

AWS:
* infra
* network security
* configuration and vulnerability analysis
* compliance validation

IAM:
* Creating users, groups, roles, policies management and monitoring
* Enabling MFA on all accounts
* Rotating keys
* Permissions management
* Analyze access patterns and review permissions

Access to Cost and Billing Console is denied even to Admin User.



[[Intro to Computing using AWS]]

All IAM principles must be authenticated to send API requests. A principal is a person or application that makes requests for an action.

Two components:
* Authentication (identity)
* Authorization (access level)
	* Identity based policy
		* user
		* group
		* role
	* Resource based policy
		* resource (such as S3 bucket)
* 