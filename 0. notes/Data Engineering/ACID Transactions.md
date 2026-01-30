#data-engineering [[Data Lakes]]

* Atomicity
	* Every update in a transaction is treated as a single unit. 
	* If any update fails then the whole transaction fails and all the updates that had been applied are rolled back.
* Consistency 
	* Reads and writes are always consistent.
	* Any queries running against a table must either get results before the update started or as the table is after the update is completed and committed.
* Isolation 
	* Transactions are isolated from each other.
	* This means that updates don't interfere with each other.
	* Transactions are queued and executed sequentially.
* Durability 
	* Once an update is applied, it is permanent.
	* All future reads must reflect the new state of the data.



