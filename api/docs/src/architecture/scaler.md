# Scaler
---
Thorium scalers are responsible for determining when and where reactions/jobs
are spawned. It accomplishes this by crawling the deadline stream and based
on fair share scheduling logic. This means that some portion of your cluster
will be dedicated to the most pressing jobs based on deadline while another
portion will be trying to fairly executing everyone's jobs evenly. This allows
for users to spawn large groups of reactions/jobs without fear of abusing
the cluster and preventing others from accomplishing tasks.

The scaler currently support 3 scheduling targets:
  - Kubernetes
  - Bare metal
  - Windows

# Scheduling Algorithms
---
The scaler uses a pool based scheduling system where each pool has its own
resources that are allocated based on their own scheduling algorithm. The
current pools in Thorium are:
  - Deadline
  - Fair share

### Deadline Pool
The deadline pool is scheduled in a first come first serve basis based on the
deadline set by the SLA for specific images. This means that jobs earlier in
the deadline stream will get priority over jobs later in the queue. It is
intended to ensure that some portion of the cluster is always working to meet
the SLA for all jobs. A downside of this is that heavy users can cause other
users jobs to be stuck in the created state for a long period of time.

### Fair Share Pool
The fair share pool is intended to balance resources across users, not images,
resulting in responsive execution of jobs even when heavy users are active.
This is accomplished by the scaler scoring users based on their currently
active jobs across all pools. The score increased is based on the resources
required for their currently active jobs. When scheduling jobs for the fair
share pool the users with the lowest score will get the highest priority.

workers that are spawned in the fair share pool will have a limited lifetime
depending on their original lifetime settings.

| original | under fair share |
| -------- | ---------------- |
| None | Can claim new jobs for 60 seconds before terminating |
| Time Limited | Can claim new jobs for up to 60 (or a lower time specified limit) seconds before terminating |
| Job Limited | Can claim a single job |

This limit is in place to ensure workers spawned under fair share churn often
to allow for resources to be shared across users with minimal thrashing.

# Scaler FAQ's
---

### Why do we only preempt pods when we are above 90% load
This is to prevent us from wasting resources spinning pods down when we have 
free resources still. If there are no jobs for that stage it will spin itself
down but if we have free resources then allowing them to continue to execute jobs
lowers the amount of orphaned jobs.

### Does Thorium's scaler hold the full docker image in its cache?
No the Thorium scaler doesn't download or have the full image at any point. It does
however contain metadata about docker images. This is what allows the scaler to
override the original entrypoint/command while passing that info to the agent.

### I see an External scaler what is that?
Thorium allows users to build their own scaler and purely use it as a job/file
metadata store. To do this you will set your images to use the External scaler.
