# Listing Reactions

Reactions can be listed at the pipeline level. You can list all reactions in a 
pipeline by sending a GET request to this endpoint:
```
<api_url>/reactions/list/:group/:pipeline
```

This will then return you a list of all reactions that are in that group/pipeline. The
cursor field is optional and will not exist if there are no more reactions to
return. The response should be in this format:
```json
{
  "cursor": 1,
  "names": [
    "77ee562f-8183-4683-a287-cc2c44c38790",
    "ec90622a-09a7-4b6a-b2cb-6b63d2b10e70",
    "13bf52ff-33a3-4fd2-9cd7-53037e01365a",
    "7b2fbc3c-3182-4afb-ab95-3f4a8969fbaf",
    "a25c140a-933e-4af7-b127-eac87872022e",
    "a7aa50f1-1993-446a-8af8-3cc4e377b046",
    "e8d0a22b-717c-4b90-8d01-5a85d19e673c",
    "d0b61d4e-f97d-4932-b095-6dd3926b9ea2"
  ]
}
```

You can also list reaction details with a GET request to the following endpoint.
```
<api_url>/reactions/list/:group/:pipeline/details
```

Like listing reaction names the cursor field is optional and will not exist if
there are no more reaction details to return. The response should look something
like the following:

```json
{
  "cursor": 1,
  "details": [
    {
      "id": "57f42995-77d3-42d9-9e23-43792ec7b7ac",
      "group": "TestGroupPleaseIgnore",
      "creator": "bob",
      "pipeline": "CornFarmer",
      "status": "completed",
      "current_stage": 2,
      "current_stage_progress": 1,
      "current_stage_length": 1,
      "args": {
        "planter": {
          "positionals": [],
          "kwargs": {
            "--corn": "sweet",
          },
          "switches": [],
          "opts": {
            "override_positionals": false,
            "override_kwargs": false,
            "override_cmd": null
          }
        },
        "harvester": {
          "positionals": [],
          "kwargs": {
            "--combine": "green",
          },
          "switches": ["--harvest"],
          "opts": {
            "override_positionals": false,
            "override_kwargs": false,
            "override_cmd": null
          }
        }
      },
      "priority": 1,
      "sla": "2020-09-23T09:00:51.001712252Z"
    }
  ]
}
```

You can also list reactions by both tags and statuses but you cannot list by
both. This is largely due to the fact that we solely use Redis to store data.
This means that we list data by crawling lists by a key. So we would need a 
list for every single combination of keys.

You can use the following urls to list by tags or statuses:

```
# list tags
<api_url>/reactions/tag/:group/:tag/
<api_url>/reactions/tag/:group/:tag/details
# list statuses
<api_url>/reactions/status/:group/:pipeline/:status
<api_url>/reactions/status/:group/:pipeline/:status/details
```

Tags can be used to list reactions across pipelines while status lists are
restricted to a specific pipeline.
