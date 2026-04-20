# Bans

> If you're a Thorium admin looking for instructions on adding/removing bans, see [Ban Things in Thorium](../admins/bans_admins.md).

"Bans" in Thorium are applied to entities that are misconfigured or noncompliant such that they cannot "function"
(e.g. an image cannot be scheduled, a pipeline cannot be run). As entities can have multiple bans, entities are effectively
"banned" when they have one or more bans and are "unbanned" when all their bans are resolved/removed.

Bans work hand-in-hand with [Notifications](./notifications.md) to inform developers why their tools cannot be
run. If an image/tool is banned, a notification is automatically created to explain the reasoning behind the
ban. Most bans are applied automatically by the API or scaler, but Thorium admins can also "ban" (or perhaps more
accurately, "indefinitely disable") tools/pipelines at their own discretion and provide a reason to the developer.

## How Do I Know When Something's Banned?

Let's say we're trying to run a pipeline called `harvest` in the `corn` group, but it's been banned for some reason.
When we try to run `harvest`, we'll get an error similar to the following:

```Bash
Error: Unable to create reactions: Code: 400 Bad Request Error:
{"error":"Unable to create reaction(s)! The following pipelines have
one or more bans: '[\"corn:harvest\"]'. See their notifications for details."}
```

The error instructs us to check the pipeline's notifications for details on the ban(s). We can do that using Thorctl:

```
thorctl pipelines notifications get corn harvest

[2024-10-31 22:13:00.800 UTC] ERROR: The image 'sow' has one or more bans! See the image's details for more info.
[2024-10-31 22:30:52.940 UTC] ERROR: The image 'water' has one or more bans! See the image's details for more info.
```

We got two notifications explaining that the `sow` and `water` images in our pipeline were banned. We can view their notifications
with Thorctl as well:

```
thorctl images notifications get corn sow

[2024-10-31 22:13:00.800 UTC] ERROR: Please decrease
your memory resources requirements to 64Gi maximum


thorctl images notifications get corn water

[2024-10-31 22:30:52.940 UTC] ERROR: The image volume 'corn-vol'
has a host path of '/mnt/corn-vol' that is not on the list of allowed host
paths! Ask an admin to add it to the allowed list or pick an allowed host path.
```

It looks like `sow` has a ban likely manually created by an admin instructing us to decrease the image's resource
requirements. Meanwhile, `water` has a host path volume with a mount not on the allowed list. Once we address this issues
and inform a Thorium admin, the bans will be lifted and we can again use our pipeline.

### Viewing Bans in an Entity's Metadata

A ban's notification should contain all the relevant info regarding a ban, but you can also see the ban
itself in the affected entity's metadata. You can view an entity's bans together with its metadata by
using the entity's respective `describe` command in Thorctl. By default `describe` prints a
human-readable summary that lists any bans. To get the full raw data, including the complete ban
records, pass `--format json`. For images, you would run:

```Bash
thorctl images describe <IMAGE> --format json
```

This will output the image's data in JSON format, including the image's bans:

```JSON
{
    "group": "<GROUP>",
    "name": "<IMAGE>",
    "creator": "<USER>",
    ...
    "bans": {
        "bfe49500-dfcb-4790-a6b3-379114222426": {
            "id": "bfe49500-dfcb-4790-a6b3-379114222426",
            "time_banned": "2024-10-31T22:31:59.251188Z",
            "ban_kind": {
                "Generic": {
                    "msg": "This is an example ban"
                }
            }
        }
    }
}
```

> Bans/notifications are currently not viewable in the Web UI, but this feature is planned for a future release of Thorium!

## Ban Types

Below are descriptions of the entities that can be banned, the types of bans they can receive, and what to
do to lift the ban.

### Image Bans

Image bans are applied when an image is misconfigured in some way. The image will not be scaled until the issue
is resolved.

The types of image bans are described below.

#### Invalid Host Path

An invalid host path image ban is applied when an image has an improperly configured host path volume.

Thorium admins can specify a list of paths that developers can mount to their images as a host path volume (see
the [Kubernetes docs on host paths](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath) for more details).
This list of allowed paths is called the `Host Path Whitelist`. If an admin removes a path from the whitelist that was
previously allowed, any images that configured host path volumes with that path will be automatically banned.

The ban (and associated notification) will contain the name of the offending volume and its path so developers can
quickly reconfigure their images. Removing or reconfiguring the problematic volume will automatically lift the ban.
Images with multiple invalid host path volumes will have multiple bans, one for each invalid host path.

#### Invalid Image URL

> ⚠️ This ban type is not yet implemented! It will be applied in a future release of Thorium.

An invalid image URL ban is applied when an image has an improperly configured URL. If the scaler fails to pull
an image at its configured URL multiple times, it will automatically apply a ban for an invalid URL.

The ban's description and associated notification will contain the invalid URL that led to the error. The ban is
removed once the developer modifies the image's URL, at which point the scaler will attempt to pull from the
new URL (applying a new pan if the new URL is also invalid).

#### Generic

A generic image ban is applied to an image if no other image ban type is applicable or if an admin applied the ban
manually for any arbitrary reason.

Generic bans must contain a description detailing the reason for the ban which can be found in the ban's
associated notification. Generic bans must be manually removed by a Thorium admin.

### Pipeline Bans

Pipeline bans restrict entire pipelines from being run. Rather than banning at the scaler as with image
bans, pipeline bans apply at the API and prevent reactions with the banned pipeline from being created
in the first place. The API responds to the reaction creation request with an error containing the reason
the pipeline was banned.

The types of pipeline bans are described below.

#### Invalid Image

An invalid image pipeline bans is applied when a pipeline has one or more images that are banned. This is the
most common type of a pipeline ban.

Pipeline bans for invalid images and their associated notifications will have the name of the offending image.
Resolving the image's ban(s) or removing the image from the pipeline will automatically lift the ban. Pipelines
with multiple banned images will have multiple bans, one for each banned image.

#### Generic

A generic pipeline bans is applied if no other pipeline ban type is applicable or if an admin applied the
ban manually for any arbitrary reason.

Generic bans must contain a description detailing the reason for the ban which can be found in the ban's
associated notification. Generic bans must be manually removed by a Thorium admin.
