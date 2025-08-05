# More on Children (Samples)

A sample submitted to Thorium as a result of running a Thorium reaction on another sample (the so-called "parent") is called a "child."

## Origin Metadata for Children

Like any sample, children can have [Origins](../users/uploading_files.md#file-origins) to help identify where it came from later.
Because children are submitted by the Thorium Agent automatically once a tool completes, it's the tool's responsibility to encode
origin information by placing children in the **origins' respective directories** (see the table in
[Output Collection](./configuring_images.md#output-collection)) for the Agent to collect from.

In most cases, the Agent can infer origin metadata just from the placement of children by origin directory as well as from context
on how the Agent was run (e.g. which tool is running on which sample/repo). For example, the Agent can submit children with the `Source`
origin by collecting them from the source children directory (`/tmp/thorium/children/source/` by default) and can infer metadata for
the `Source` origin – namely parent repo, commitish, flags, build system, etc. – just from the context of how the tool was run.

There are some cases, however, where the Agent cannot infer origin metadata beyond the origin type. These cases are detailed below.

### Carved from PCAP

Thorium can save a lot of useful metadata about files carved from a PCAP (packet capture) sample beyond custom tags
(see [PCAP Origin](../users/uploading_files.md#pcap) for what kind of metadata can be saved). When manually uploading samples, it's
easy to add this information in the Web UI or Thorctl. When the Thorium Agent uploads children files, though, it needs a place
to look to grab this information your tool may have extracted.

The special place the Thorium Agent looks is in the `thorium_pcap_metadata.json` file in the `CarvedPCAP` origin sub-directory
(`/tmp/thorium/children/carved/pcap/` by default). This file should be a JSON map where the keys are children filenames (*not* absolute
paths) and the values are the metadata to encode. An example `thorium_pcap_metadata.json` file could look like:

```JSON
{
    "carved_from_pcap1.html": {
        "src": "1.1.1.1",
        "dest": "2.2.2.2",
        "src_port": 80,
        "dest_port": 34250,
        "proto": "TCP",
        "url": "example.com"
    },
    "carved_from_pcap2.txt": {
        "src": "3.3.3.3",
        "dest": "4.4.4.4"
    }
}
```

The table in [PCAP Origin](../users/uploading_files.md#pcap) lists the fields each child file may have. The `thorium_pcap_metadata.json` file
is completely optional. If no metadata file is provided, all PCAP-carved children will still have the `CarvedPCAP` origin, just with no
metadata beyond the parent SHA256, the tool that carved out the file, as well as any custom tags your tool sets.
