# Generators
---

Generators allow developers to programmatically spawn pipelines in Thorium. This
means a pipeline can behave like an event loop spawning reactions and then
acting on their results or some other events. An example of this would be a
pipeline that lists data in Thorium and spawns reactions for each item.

### Generators Lifecycle
---
The lifetime of generators in Thorium can be shown by the following flowchart:

<p align="center">
    <img width="600" src="./../static_resources/developers/generator-lifecycle.svg">
</p>

Each time the generator sleeps it will not be rescheduled until all the
sub-reactions it spawned reach a terminal state (completed/failed). When it is
respawned it will be given the checkpoint info it set previously. This allows
it to pick back up where it left off. When spawning sub-reactions it is highly
recommended to spawn a limited number of sub-reactions each loop. This number
depends on how long the target image pipeline takes to complete but 500-1000 is
a good rule of thumb.

#### Sleep/Respawn

In order to respawn after its sub-reactions are complete, the generator must
signal to Thorium that it should be put in a sleeping state before exiting. If
the generator exits without sending the sleep request, Thorium will finish
the generator job and refrain from respawning it.

You can tell Thorium to sleep a generator by POSTing to this URL:

```
<API_URL>/reactions/handle/:job_id/sleep?checkpoint=<checkpoint>
```

The generator receives its `job_id` from Thorium from the `--job` kwarg.

#### Checkpoints

A checkpoint is a custom string that can be given to a generator to give it
context from its previous run. Checkpoints are passed to the reaction with the
`--checkpoint` kwarg.

For example, a generator might spawn 50 reactions and then send a sleep request
with the checkpoint `"50"`. When the generator respawns, it will be run with the
kwarg `--checkpoint 50`. This way, the generator can keep a running count for
how many sub-reactions it has spawned. Checkpoints can also be used to simply
signal to the generator that it's been respawned at all.

### Example

If we extend previous example with the following requirements:
 - List files in Thorium
 - Spawn the Inspector image on files tagged with ```Submitter=mcarson```

Then our generators logic would look like:

<p align="center">
    <img width="600" src="./../static_resources/developers/generator-lifecycle-example.svg">
</p>


### FAQ
---
##### When will my generator get respawned?

Generators are respawned when all of the sub-reactions they created reach a final
state (completed or error).

##### Why should generators limit how many jobs they create per generation loop?

Drip feeding jobs into Thorium instead of adding them all at once lowers the
burden on the Thorium scheduler by avoiding creating millions of jobs at a
time.

##### How do I get reaction/job ID's for my generator?

Thorium will pass the generator's reaction/job ID's to the generator with the
`--job`/`--reaction` kwargs, respectively.
