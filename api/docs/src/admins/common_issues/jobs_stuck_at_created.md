# Jobs Stuck In Created State

When jobs are stuck in the created state for an extended period of time this
can be due to multiple issues:
- High load
- Outdated agent
- Missing Volumes

### High Load
---
When the cluster has a high number of jobs in queue, jobs may be
in a created state for an extended period of time. The fair share scheduler
should help mitigate this when other users are the cause of the load (the fair
share scheduler balances across users, not images/pipelines). If the user
experiencing the stuck jobs is also the cause of heavy load, the user needs
to wait for their in-progress jobs to complete before their other jobs can
be scheduled.

### Outdated Agents
---
A common issue for jobs being stuck in the created state after updating Thorium
is the agent failed to update. Before the agent claims any job it will
check the version of the API against its own. If it is the incorrect version
then the agent will exit without claiming a job.

#### Getting the current version
In order to get the current api version run the following command:

<script>
  let base = window.location.origin;
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("curl " + base + "/api/version");
  document.write("</code>");
  document.write("</pre>");
</script>

#### Outdated Agents: Kubernetes

In order to determine if the agent version is incorrect on kubernetes, first get
pod logs with the following:

```
kubectl get pods -n <NAMESPACE> | grep "/1" | awk '{print $1}' | kubectl logs -n <Namespace> -f
```

If any of the logs show the agent exiting without claiming a job due to
version mismatch, run the following command to update the Thorium
agent on all nodes.

```
kubectl rollout restart deployment operator -n thorium
```

#### Outdated Agents: Bare Metal

On bare metal machines the agent is auto updated by the Thorium reactor. To
confirm if the version is correct, simply run the following command to check the
reactor:
```
/opt/thorium/thorium-reactor -V
```

Then to check the agent, run the following:
```
/opt/thorium/thorium-agent -V
```

In order to update the reactor, run the following command:

<script>
  let hostname = window.location.hostname;
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("curl " + hostname + ":39080/api/binaries/linux/x86-64/thorium-reactor -o thorium-reactor");
  document.write("</code>");
  document.write("</pre>");
</script>

In order to update the agent, run the following command:

<script>
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("curl " + hostname + ":39080/api/binaries/linux/x86-64/thorium-agent -o thorium-agent");
  document.write("</code>");
  document.write("</pre>");
</script>

### Missing Volumes

Another common issue that can cause K8s-based workers to get stuck in the
created state is missing volumes. This occurs when the user has defined their image
to require a volume, but the volume has not been created in K8s. This causes
the pod to be stuck in the ContainerCreating state in K8s. To get pods in
this state run the following command

```
kubectl get pods -n <NAMESPACE> | grep "ContainerCreating"
```

Then looks to see if any pods have been in that state for an extended period of
time by checking the age of the pod. For example, this pod has been stuck for 10
minutes and is likely missing a volume.

```
➜ ~ kubectl get pods
NAME       READY   STATUS              RESTARTS   AGE
underwater-basketweaver-njs8smrl 0/1     ContainerCreating   0          10m
```

To confirm this is the issue describe the pod with the following command and
check the events:

```
kubectl describe pod/<POD> -n <NAMESPACE>
```

If there is an event similar to the following, you are missing a volume
that needs to be created.


```
Events:
  Type     Reason       Age                   From               Message
  ----     ------       ----                  ----               -------
  Normal   Scheduled    10m45s                 default-scheduler  Successfully assigned <NAMESPACE>/<POD> to <NODE>
  Warning  FailedMount  51s (x12 over 10m45s)  kubelet            MountVolume.SetUp failed for volume "important-volume" : configmap "important-volume" not found
````
