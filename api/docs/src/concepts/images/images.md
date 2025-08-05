# Images

Images are used to define what each stage of a pipeline looks like. Each stage
can have multiple images or a single image. Separating the image declarations
from pipeline declaration allows you to reuse images across pipelines
without having to redefine an image every time. This also makes updating
images easier as their is less duplicate information to update.

The first step to creating a pipeline is to define the images that your pipeline
will leverage. You do this by POSTing image definitions to the api at
```
<api_url>/images/  
```

An image definition is submitted in json format and something like this
```json
{
    "group": "petshop",
    "name": "adopter",
    "external": false,
    "image": "petshop:adopter",
    "env": {
      "petshop_name": "Thorium Pet Shop",
      "http_proxy": null,
      "https_proxy": null,
      "HTTP_PROXY": null,
      "HTTPS_PROXY": null
    },
    "lifetime": {
      "counter": "jobs",
      "amount": 1
    },
    "timeout": 300,
    "requests": {
      "cpu": "4",
      "memory": "3Gi"
    },
    "limits": {
      "cpu": "6",
      "memory": "4Gi"
    },
    "volumes": [
      {
        "name": "petshop-secrets",
        "archetype": "Secret",
        "mount_path": "/confs/petshop-secrets.yml",
        "sub_path": "petshop-secrets.yml"
      },
      {
        "name": "petshop-configs",
        "archetype": "ConfigMap",
        "mount_path": "/confs/petshop-configs.yml",
        "sub_path": "petshop-configs.yml"
      }
    ],
    "modifiers": "/tmp/thorium/mods",
    "security_context": {
      "user": 1234,
      "group": "5678",
      "allow_privilege_escalation": false,
    },
    "collect_logs": true,
    "generator": false,
}
```

This json blob contains all of the information necessary for Thorium to
use this image in a pipeline. Explanations for the fields are:

| key | definition |
| --- | ---------- |
| group | The name of the group this image is in |
| name | The name of pipelines should use when referring to this image |
| external | Whether Thorium is responsible for spawning workers for this image (optional) |
| image | The docker image to launch (optional) |
| env | The environment variables to set inside the container |
| lifetime | The lifetime of the spawned pod (optional) |
| timeout | The max time a single job can run for in seconds (optional) |
| requests | The minimum amount of resources required for this image (optional) |
| limits | The maximum amount of resources to give this image (optional) |
| volumes | The volumes this image requires to be spawned (optional) |
| modifiers | The path the Thorium agent should look for reaction modification files (optional) |
| security_context | The security settings Thorium should enforce on pods using this image (optional) |
| collect_logs | Whether to stream logs back to the API, defaults to true (optional) |
| generator | Whether this image is a generator or not (optional) |

Most of the above fields are optional depending on if other fields are set.
The first of these fields is external. This determines whether Thorium is responsible
for spawning your workers or not. If external is set to true then the Thorium
scaler will ignore those jobs entirely. If you set external to true then you
should not set the image, lifetime, requests, or limits fields.

#### External
---
If external is set to false then you must set the image and requests fields.
Lifetime and limits are still optional.

#### Requests & Limits
---
Requests and limits are very similar, but represent the minimum and maximum
bounds on the amount of resources a image may consume. This means that if you
set your requested resources to be 16 cpu cores your image will only be spawned
if the cluster has at minimum 16 cores.
The shared keys to define these are:

| key | definition |
| --- | ---------- |
| cpu | The number of cpu cores |
| memory | The amount of memory to give this pod (must have label) |

Some keys can only be defined in the limits block. One of these keys is the gpu
keys. This can only be defined in the limits block and it will be ignored in the
requests block. You can request GPUs with the following keys:

| type | key |
| --- | ---------- |
| AMD | amd.com/gpu |
| Nvidia | nvidia.com/gpu |

#### Lifetime
---
Lifetime is how long your pod will live if we assume there is a constant stream
of jobs (because Thorium pods die automatically if no jobs exist). There are two
types of lifetime handlers currently:

| handler | description |
| --- | --------- |
| jobs | determine lifetime based on jobs claimed |
| time | determine lifetime based on time alive (in seconds) |

Both of these lifetime handlers are not strongly enforced. This means that it 
is not guaranteed that pods will not outlive their lifetime. This is because
lifetime is checked in between every loop when claiming jobs. So if we have an
image that attempts to claim and execute N jobs then it is possible to execute
at most N - 1 extra jobs before the lifetime handler catches it. A similar
situation exists with the time handler but it is less defined as it depends on
the time it takes to run a job and when that job is claimed in relation to the
lifetime expiration.

#### Timeout 
---
Timeout is similar to lifetime but it only constrains how long an individual job
can run for. This means that if a timeout of 60 is set any job for that image will
error out if it runs for longer then 60 seconds. This is enforced with a max
resolution of 100ms. So a job may execute for 60.1 seconds and still complete.

#### Volumes 
---
Volumes are how you tell Thorium that pods based on this image need to have a 
volume bound in when being spawned. Currently Thorium supports three volume types

- ConfigMap
- Secret
- NFS

For more information, see [Volumes in Images](./volumes.md).

#### Modifiers 
---
The modifiers path is where the Thorium agent should look for files dropped while
this stage was executing that tell Thorium how it should modify the remaining stages
of this reaction.

For more information, see [Modifiers](./../reactions/modifiers.md).

#### Security Context 
---
A security context is the settings Thorium uses to restrict the pods for this image.
They allow you to designate what user and group that process spawned in that pod should
have and to disallow privilege escalation. All fields in this are optional and privilege
escalation defaults to false.

#### Runtime 
---
Later when listing images you may notice a field called runtime is created. This
is how long on average your image takes to execute a job. Thorium uses this value
internally to determine when to execute your jobs vs other jobs. This value will
update over time and get more accurate as you run more jobs.

### Generator
---
A Generator is an image that crawls over some input or external data source creating
sub reactions. It will create a batch of jobs (typically no more than 1-5,000) and
then tell Thorium it should be slept before exiting with a return code of 0. If a
generator stage completes without telling Thorium it should be slept the generator
stage will be completed and no longer will be recreated. When telling a generator to
sleep you can also pass in a checkpoint string. This will be then passed in as a
keyword arg to your job as --checkpoint allowing your generator to resume progress.

You can tell Thorium to sleep a generator by POSTing to this URL

```
<API_URL>/reactions/handle/:job_id/sleep?checkpoint=<checkpoint>
```

Thorium will pass the generator image the reaction/job ID's with the `--reaction`/`--job`
kwargs, respectively.

Generators are preferred over just creating millions of jobs at once due to how that
impacts scheduling. Because the Thorium scheduler cannot easily view all jobs in the
deadline stream at once (as that would require us to download the entire stream each
scale loop) it only looks at the next 100k jobs in the stream. This means that when
doing fair share scheduling if you are the 100,001th job you will not be scheduled under
fair share in this scale loop. By drip feeding jobs into Thorium though we can keep
the deadline stream smaller and more manageable without requiring human interaction.
