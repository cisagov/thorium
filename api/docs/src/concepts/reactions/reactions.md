# Reactions

Reactions are the execution of a single pipeline and are what allow you to
follow the jobs that make up your pipeline. Reactions are used over tracking the 
individual job IDs as those constantly change as stages are completed. This 
allows you to track the execution of a pipeline without needing to keep track
of all the jobs within that pipeline.

This also allows your images to be completely ignorant of how and where they
fit into those pipelines. They do not need to know how to call or create a job
for the next stage they simply need to be able to store their results somewhere
they can be accessed. This makes it extremely easy to quickly tie analytics
together.

Reactions can be created by POSTing to the following endpoint:
```
<api_url>/reactions/
```

A reaction declaration looks something like this:
```json
{
  "group": "petshop",
  "pipeline": "adopter",
  "args": {
    "adopt": {
      "positionals": ["1"],
      "kwargs": {
        "--color": "yellow",
      },
      "switches": ["--puppy"]
    }
  },
  "sla": 86400,
  "parent": "4d08deaa-eb23-4519-9719-cc216083d692"
}
```

Explanations for this field are:

| key | definition |
| --- | ---------- |
| group | The group the pipeline to create a reaction for is in |
| pipeline | The pipeline to create a reaction for |
| args | The reaction argument struct (explained below) |
| sla | How long you can wait for this reaction to complete in seconds (optional) |
| parent | The parent reaction for this sub reaction (optional) |

### Arguments
---
Reaction arguments are a hashmap of arguments to overlay on top of the docker
file for each image in the pipeline. So for example in the reaction above for a
petshop we have a single image called adopt that specifies arguments to adopt a
yellow puppy. If we assume the original docker file had its cmd set to:

```
["--color", "brown"]
```

Because we specified an arg to overlay on top of "--color" that would be set to
"yellow" when we call the entrypoint instead of "brown". This also means that 
if you simply want to use the args specified in the docker file every time you
can leave args set to an empty {}. You also do not need to specify args for
every image in the pipeline.

You can also specify to completely override the original args in the docker file
as well.

```json
{
  "group": "petshop",
  "pipeline": "adopter",
  "args": {
    "adopt": {
      "opts" : {
        "override_positionals": true,
        "override_kwargs": true,
        "override_cmd": ["./adopt", "1", "--color", "black", "--kitten"]
      }
    }
  },
  "sla": 86400
}
```

### SLA
---
This is how long in seconds you can wait for this reaction to complete. It is
important to remember that Thorium tries its best to meet this SLA but it is not
a guarantee. Thorium uses this value to determine priority and when it should
schedule your reaction relative to others. If no SLA is given then it defaults to
the SLA given to the pipeline and if that has no SLA then 1 week is used.

### Sub Reactions
---

Sub Reactions are a way to dynamically spawn tasks that block the spawning of
later stages of your current reaction until all sub reactions complete. This
allows users to dynamically spawn tasks instead of having to encode all
dependencies into their pipeline statically. Sub reactions however do not fail
out their parent reaction if they fail. This means that if a parent reaction
cannot complete without a sub reaction completing the onus of ensuring that
is entirely dependent on the creator of the sub reactions. This was done to
more easily support Generator logic when it comes to sub reactions.

Creating a sub reaction is exactly the same as creating a regular reaction but
you set a parent field to your current reactions ID.

### Creating Reactions in bulk
---
You can also create reactions in bulk in a single request in order to efficiently
create large amounts of reactions. This request looks almost exactly the same as
when creating a single reaction but instead the reaction requests are in a list.

Reactions can be created in bulk by POSTing to the following endpoint:
```
<api_url>/reactions/bulk
```

The body should look something like:

```json
[
  {
    "group": "petshop",
    "pipeline": "adopter",
    "args": {
      "adopt": {
        "positionals": ["1"],
        "kwargs": {
          "--color": "yellow",
        },
        "switches": ["--puppy"]
      }
    },
    "sla": 86400
  },
  {
    "group": "petshop",
    "pipeline": "adopter",
    "args": {
      "adopt": {
        "positionals": ["1"],
        "kwargs": {
          "--color": "black",
        },
        "switches": ["--kitten"]
      }
    },
    "sla": 86400
  }
]
```
