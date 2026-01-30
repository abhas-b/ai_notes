#aws [[AWS 1|AWS 1]]

1. Can be used for sending job related notifications with [[Glue]]
2. Types of topics:
	1. FIFO
		1. preserve message ordering
		2. exactly once message delivery
	2. Standard
		1. Best effort message ordering
		2. at least once message delivery
3. We need to subscribe to a topic to publish notifications.
4. 