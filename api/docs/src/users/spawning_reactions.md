# Spawning Reactions

In Thorium terminology, a `reaction` is a unit of work where one or more tools run on some data within a `pipeline`.
Thorium allows for many tools, called `images`, to be strung together into sequential or parallel `stages` of
a `pipeline`. The process for configuring images to run within Thorium and building pipelines is covered in detail
within the [Developer chapters](../developers/developers.md).

## WebUI
---
The Web UI currently only allows users to spawn reactions for a single file at a time. If you wish to spawn reactions on
many files, follow the Thorctl examples below. Once you have spawned a reaction, you can follow its progress and even
view the stdout/stderr in the logs for that reaction stage. This allows you to easily troubleshoot tools if your analysis
jobs fail to complete successfully.

<video autoplay loop controls>
  <source src="../static_resources/reactions/reaction-spawn.mp4", type="video/mp4">
</video>


## Thorctl
---
Thorctl allows you to spawn reactions for a single file or many files at once. Use the following command to spawn a single reaction
on a specific file using the file's SHA256 hash:

```bash
thorctl reactions create --group <PIPELINE_GROUP> --pipeline <PIPELINE> <SHA256>
```

If you want to run a pipeline on files that have a specific tag or tags, add the `-t/--tags` flag and specify a tag in the format
`KEY=VALUE` as shown below:
```bash
thorctl reactions create --limit <LIMIT> --group <PIPELINE_GROUP> --pipeline <PIPELINE> --tags Dataset=Examples
```

To specify multiple tags, enter a `-t/--tags` flag for each tag:
```bash
thorctl reactions create --limit <LIMIT> --group <PIPELINE_GROUP> --pipeline <PIPELINE> --tags Tag1=Hello --tags Tag2=Goodbye
```

You can also watch the status of reactions using `--watch` or `-W`.

```bash
$ thorctl reactions create --group demo --pipeline test-pipeline --watch
CODE | PIPELINE                  | SAMPLES                                                          | ID                                   | MESSAGE                         
-----+---------------------------+------------------------------------------------------------------+--------------------------------------+----------------------------------
200  | test-pipeline              | 85622c435c5d605bc0a226fa05f94db7e030403bbad56e6b6933c6b0eda06ab5 | a0498ac4-42db-4fe0-884a-e28876ec3496 | -
-----+---------------------------+------------------------------------------------------------------+--------------------------------------+----------------------------------

	WATCHING REACTIONS	

STATUS       | PIPELINE                  | ID                                  
-------------+---------------------------+--------------------------------------
...
```

### Thorctl Run
You can also quickly create a reaction, monitor its progress, and save its results to disk using the `thorctl run` command:

```bash
thorctl run <PIPELINE> <SHA256>
```

Unlike `thorctl reactions create`, `thorctl run` will display the stdout/stderr output of each stage in real time and
automatically save the results to disk, effectively emulating running the reaction locally on your machine. This might be
preferable to `thorctl reactions create` for running a quick, one-off reaction.

# Reaction Status

The status of a reaction can be used for monitoring the progress of the analysis jobs you create. You can view the
status of reactions on the file details page through the Web UI or using the `-W` flag when submitting reactions using
Thorctl. 

After a reaction has been submitted, its initial status is `Created`. Reactions that have been scheduled by the Thorium
Scaler and executed by an Agent process will enter the `Running` state. These reactions will run until either the
tool completes successfully, returns an error code, or is terminated by Thorium for exceeding its runtime specification
(resources limits or max runtime). All failure states will cause the reaction to enter the `Failed` state. Successful
runs of all images within the pipeline will cause the reaction to be marked as `Completed`.

| Status | Definition |
| ---- | ---- |
| Created | The reaction has been created but is not yet running. |
| Running | At least one stage of the reaction has started. |
| Completed | This reaction has completed successfully. |
| Failed | The reaction has failed due to an error. |

# Reaction Lifetimes

Once a reaction has reached its terminal state (`Completed` or `Failed`), the reaction status and logs will see no
future updates. Thorium applies a lifespan of 2 weeks for reactions that have reached a terminal state. After this
lifespan has been reached, Thorium will cleanup info about the expired reaction. This cleanup does not delete tool
results and only affects reaction metadata such as the reaction's status and logs. This helps to prevent infinite
growth of Thorium's high consistency in-memory database, Redis. Because of this cleanup, users may not see any
Reactions listed in the `Reaction Status` section of the Web UI file details page even when tool results are visible.