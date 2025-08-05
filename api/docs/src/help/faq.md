# Frequently Asked Questions

## Docs
---

##### Why can't I view the videos embedded in these docs?

The videos in these docs are AV1 encoded. Since version 116, the Edge browser for Windows does not come with a builtin
plugin for viewing AV1 formatted video. Instead you need to search for an add-on extension via Microsoft's website.
Most other browsers such as Chrome, come with AV1 support by default.


## Data
---

#### What data types can be uploaded to Thorium?

Thorium is primarily a file analysis and data generation platform. As such, it supports two primary types of data:

  - files
  - repositories (Git)

There are no restrictions on the file types that Thorium supports. All files are treated like raw data and
safely packaged using CaRT upon upload. Some commonly uploaded file formats include binary executables (PEs, ELFs,
etc.), library files (DLLs), archives (zips), office documents (PDFs) and many more. Repositories are a separate
data type that can also be ingested into Thorium and comes with some additional features that enable building of
versioned binaries from a large number of repos at scale.

#### What is CaRT and how can I unCaRT malware samples that I download from Thorium?

CaRT is a file format for the safe and secure transfer of malware samples and was developed by Canada's CSE. CaRTed
files are neutered and encrypted to prevent accidental execution or quarantine by antivirus software when downloaded
from Thorium. All files are CaRTed by the Thorium API upon upload and must be unCaRTed by the user after they are
downloaded. You can use the Thorium CLI tool (Thorctl) to unCaRT your downloaded file. For more info about Thorctl
see our setup [instructions](../architecture/thorctl.md).


## Tools
---

#### How can I add my own tools and build pipelines in Thorium?

Thorium has been designed to support quickly adding new tools and building pipelines from those tools. Tools do not
need to understand how to communicate with the Thorium API or the CaRT file storage format. Any command line tool that
can be configured to run within a container or on BareMetal can be run by Thorium. You can read more about the process
for adding tools and pipelines in the [developer docs](../developers/developers.md).

## Sharing and Permissions
---

#### How can I share or limit sharing of the data I upload to Thorium?

All data is uploaded to a group and only people within that group can see that group's data. If you want to share data
with someone, you can add that person to the group or reupload that data to one of their groups. You can read about how
Thorium manages data access with groups and how group and system roles affect the ability of users to work with Thorium
resources in the [Roles and Permissions](../getting_started/roles_permissions.md#group-ownership) section of the
[Getting Started](../getting_started/getting_started.md) chapter.

#### What is Traffic Light Protocol (TLP) and does Thorium support TLP levels for files?

> TLP provides a simple and intuitive schema for indicating when and how sensitive information can be shared,
facilitating more frequent and effective collaboration. - [https://www.cisa.gov/tlp](https://www.cisa.gov/tlp)

The Thorium Web UI supports tagging uploaded files with a TLP metadata tag. This tag is treated just like any other
tag that is applied to a new or existing file. If the TLP level changes, a Thorium user with the correct permissions
can modify that TLP tag in order to ensure it is kept up-to-date.
