# Viewing Results

Tool results are created when a `pipeline` is run on a target `file`. The running `pipeline` instance is called a
`reaction` and may involve running several tools (called `images`) on the target `file`. The analysis artifacts created
by each tool are then automatically stored in Thorium after each `pipeline` stage has completed. We organize tool
results based on the name of the tool/image rather than the name of the pipeline where that tool was run.

Tools may generate several types of result output, including renderable and downloadable formats. These artifacts
include:

- results: renderable data including basic text and JSON formatted tool output
- result-files: downloadable files produced by the tool and any tool results larger than 1MB
- children: unpacked or transformed files that Thorium treats like sample files due to potential maliciousness.

You can view or download results and child files from both the Web UI and Thorctl.

## Web UI
---

### Results and Result Files

You can navigate to the details page for a file using the `sha256` hash of that file, or by browsing and searching
through Thorium's data. If you are already on the file details page and see your reaction's have completed, refresh the
page to get the latest tool results!

<script>
  let base = window.location.origin;
  document.write("<pre>");
  document.write("<a href=" + base + "/file/SHA256" + ">");
  document.write(base + "/file/[SHA256]");
  document.write("</a>");
  document.write("</pre>");
</script>

Once you load the file details page, click the `Results` tab that's displayed after the submission info section. You
should see tool results that you can scroll through as shown in the video below.

<video autoplay loop controls>
  <source src="../static_resources/results/results-view.mp4", type="video/mp4">
</video>

You can also jump to results by clicking on the tools corresponding tag for a particular tool result.

<video autoplay loop controls>
  <source src="../static_resources/results/results-jump-by-tag.mp4", type="video/mp4">
</video>

Tools can create renderable results as well as result files. If a tool produces a result files, those files can be
downloaded using the links at the bottom of the result section for that tool.

<video autoplay loop controls>
  <source src="../static_resources/results/results-files-download.mp4", type="video/mp4">
</video>

The number of result files that a tool produced will be displayed on the header of the results section. That file
count badge can be clicked to jump to the result files links.

<video autoplay loop controls>
  <source src="../static_resources/results/results-files-view.mp4", type="video/mp4">
</video>

### Children Files

Many tools will produce entirely new samples called `children files` that are saved in Thorium after the tool exits.
For example, an unpacking tool might remove protective/obfuscating layers of a given malware sample in order to unpack
the core payload and save it as a new sample in Thorium for further analysis. The sample that a tool was run on to
produce a child file its `parent file`. The origin information on a child file's details page contains a convenient link
to the child's parent. Clicking the link will take you to the sample details of the parent file.

<p align="center">
    <img width="800" src="./../static_resources/results/children-files-parent-link.png">
</p>

## Thorctl
---

You can download results for specific samples using Thorctl with the following command:

```bash
thorctl results get <SHA256>
```

Download results for multiple samples by passing multiple file SHA's:

```bash
thorctl results get <SHA256-1> <SHA256-2> <SHA256-3>
```

If you want to download results for specific tools then you can use the following command:

```bash
thorctl results get --tools <TOOL> --tools <TOOL> <SHA256>
```

You can also get results for any samples with a certain tag with the following command:

```bash
thorctl results get --tags Dataset=Examples
```

The tool and tag flags can be set together to get the results of running a tool on samples with a particular
characteristic:

```bash
thorctl results get --tools analyzer --tags Packed=True
```

The number of results from which Thorctl downloads files is limited to prevent inadvertent massive download requests.
To change the limit, use the `--limit`/`-l` flag:

```bash
thorctl results get --tags Incident=10001234 --limit 100
```