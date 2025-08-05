# Event Handler
---
The event handler in Thorium is responsible for triggering reactions based on
events in Thorium. An event in thorium is an action taking place like:
- Uploading a file/repo
- Creating tags

When these event happen they are pushed into a stream in redis. The event handler
then pops events from this stream and determines if the conditions for a pipeline
trigger have been met. If they have then a reaction will be created for the user
whose event met this triggers conditions. A single event can trigger multiple
distinct triggers.

### Event Handler FAQ's
---

### Is there a delay between events being created and being processed
Yes, the event handler trails live events by 3 seconds. This is to ensure that
Scylla has a chance to become consistent before the event handler process an
event. Event though event data is stored in Redis the event-handler often has
to query for additional data to determine if a trigger's conditions have been
met. This data is stored in Scylla and so requires some time to become
consistent.

### What stops an infinite loop in events?
Triggers have a configurable depth limit meaning any events that reach that
limit will be immediately dropped instead of processed.

### Can I replay events?
No, once an event is processed it is dropped and cannot be replayed.
