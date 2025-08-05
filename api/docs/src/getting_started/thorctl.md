# Thorctl
---
Thorctl is a command line tool aimed at enabling large scale operations within Thorium. Thorctl provides a variety
of features including:

- uploading files
- uploading Git repositories
- ingesting Git repositories by URL
- downloading files/repos
- starting reactions/jobs
- starting Git repo builds
- downloading results
- listing files

An example of some of these can be found in the [Users](./../users/users.md)
section of these docs.

To install Thorctl, follow the instructions for your specific operating system in the sections below.

### Linux/Mac

On a Linux or Mac machine, open a terminal window and run the following command:

<script>
  let base = window.location.origin;
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("curl " + base + "/api/binaries/install-thorctl.sh | bash -s -- " + base);
  document.write("</code>");
  document.write("</pre>");
</script>

  #### Insecure Download

  - Although not recommended, you can bypass certificate validation and download Thorctl insecurely
    by adding the `-k` (insecure) flag to `curl` and `--insecure` at the very end of the command
    (see the command below for reference). The former tells `curl` to download the script itself
    insecurely while the latter will informs the script to use insecure communication when downloading
    Thorctl.

<script>
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("curl -k " + base + "/api/binaries/install-thorctl.sh | bash -s -- " + base + " --insecure");
  document.write("</code>");
  document.write("</pre>");
</script>

### Windows

Download Thorctl from the following link: [Windows Thorctl](../../../binaries/windows/x86-64/thorctl.exe)


### Login Via Thorctl

After you have downloaded Thorctl, you can authenticate by running:

<script>
  document.write("<pre>");
  document.write("<code class=\"language-bash  hljs\">");
  document.write("thorctl login " + base);
  document.write("</code>");
  document.write("</pre>");
</script>

<video autoplay loop controls>
  <source src="../static_resources/thorctl-login.mp4", type="video/mp4">
</video>

### Configure Thorctl

Logging into Thorium using `thorctl login` will generate a Thorctl config file containing the
user's authentication key and the API to authenticate to. By default, the config is stored
in `<USER-HOME-DIR>/.thorium/config.yml`, but you can manually specify a path like so:

```
thorctl --config <PATH-TO-CONFIG-FILE> ...
```

The config file can also contain various other optional Thorctl settings. To easily modify the config
file, use `thorctl config`. For example, you can disable the automatic check for Thorctl updates
by running:

```
thorctl config --skip-updates=true
```

You can specify a config file to modify using the `--config` flag as described above:

```
thorctl --config <PATH-TO-CONFIG-FILE> config --skip-updates=true
```

### Thorctl Help

Thorctl will print help info if you pass in either the `-h` or `--help` flags.

```bash
$ thorctl -h
The command line args passed to Thorctl

Usage: thorctl [OPTIONS] <COMMAND>

Commands:
  clusters   Manage Thorium clusters
  login      Login to a Thorium cluster
  files      Perform file related tasks
  reactions  Perform reactions related tasks
  results    Perform results related tasks
  repos      Perform repositories related tasks
  help       Print this message or the help of the given subcommand(s)

Options:
      --admin <ADMIN>      The path to load the core Thorium config file from for admin actions [default: ~/.thorium/thorium.yml]
      --config <CONFIG>    path to authentication key files for regular actions [default: ~/.thorium/config.yml]
  -k, --keys <KEYS>        The path to the single user auth keys to use in place of the Thorctl config
  -w, --workers <WORKERS>  The number of parallel async actions to process at once [default: 10]
  -h, --help               Print help
  -V, --version            Print version
```

Each subcommand of Thorctl (eg `files`) has its own help menu to inform users on the available options for that
subcommand.

```bash
$ thorctl files upload --help
Upload some files and/or directories to Thorium

Usage: thorctl files upload [OPTIONS] --file-groups <GROUPS> [TARGETS]...

Arguments:
  [TARGETS]...  The files and or folders to upload

Options:
  -g, --groups <GROUPS>            The groups to upload these files to
  -p, --pipelines <PIPELINES>      The pipelines to spawn for all files that are uploaded
  -t, --tags <TAGS>                The tags to add to any files uploaded where key/value is separated by a delimiter
      --delimiter <DELIMITER>      The delimiter character to use when splitting tags into key/values [default: =]
  -f, --filter <FILTER>            Any regular expressions to use to determine which files to upload
  -s, --skip <SKIP>                Any regular expressions to use to determine which files to skip
      --folder-tags <FOLDER_TAGS>  The tags keys to use for each folder name starting at the root of the specified targets
  -h, --help                       Print help
  -V, --version                    Print version
  ```
