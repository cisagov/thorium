# Downloading Files

If you need to download a file to carry out further manual analysis steps, you can do so via the Web UI or Thorctl.
Because samples stored in Thorium are often malicious, they are downloaded from Thorium in a non-executable state,
either in a safe
[CaRTed](../help/faq.html#what-is-cart-and-how-can-i-uncart-malware-samples-that-i-download-from-thorium) format
or as encrypted ZIP files. This means that before a downloaded file can be analyzed, it must be either be unCaRTed
or decrypted/extracted from the ZIP archive to return it to its original, potentially executable, state. If you are
working with malicious or potentially malicious files, only unCaRT them in a safe location such as a firewalled
virtual machine. Keep in mind that most anti-virus applications will immediately detect and quarantine known malware
after extraction, so disabling anti-virus applications entirely may be necessary to effectively extract the sample.
Be careful when dealing with extracted malware samples!

### Cart vs Encrypted Zip

Thorium supports two different download types each with its own pros and cons:

| Capability | CaRT | Encrypted Zip |
| ---------- | ---- | ------------- |
| Encrypted | ✅ | ✅ |
| Compressed | ✅ | ✅ |
| Streaming Extraction | ✅ | ❌ |
| API Load | low | high |
| Native Windows/Linux/Mac Support | ❌ | ✅ |

At a high level encrypted zips are more user friendly but are less performant
and cause a high load on the API. When performing actions at scale or working
with large files using CaRT is highly recommended.

### Web UI Sample File Download
---

You can choose between [CaRTed](../help/faq.html#what-is-cart-and-how-can-i-uncart-malware-samples-that-i-download-from-thorium)
and encrypted ZIP format when downloading files using the Thorium Web UI. If the file is in the CaRTed format You will need to use
a tool such as Thorctl to unCaRT the file after it has been downloaded and moved into a sandboxed environment.

<video autoplay loop controls>
  <source src="../static_resources/files/file-download.mp4", type="video/mp4">
</video>

### Thorctl File Download
---

Alternatively, you may use Thorctl to download the file on the command line in either a CaRTed or unCaRTed format.
You can download a single file by its sha256 hash using the following Thorctl command:

```bash
thorctl files download <sha256>
```

Thorctl's current behavior is to download the file in a CaRTed format by default. Downloading files as encrypted ZIP's is not
currently supported in Thorctl. If you want to immediately unCaRT the file, you can use the `-u` or `--uncarted` flag.

```bash
thorctl files download --uncarted <sha256>
```

If you want to download the file to a different path, that is not in the current working directory, you can use the
`-o/--output` flag.

```bash
thorctl files download --output /path/to/download/directory <sha256>
```

You can also download multiple files by specifying a metadata tag that the downloaded files must have and the `-l/--limit` flag
to specify how many files you would like to download.

```bash
thorctl files download --carted --limit 100 --tags Incident=10001234
```

If you do not specify a limit count when you provide a key/value tag, Thorctl will default to downloading a maximum of 10 files.

### CaRTing/UnCaRTing Files
---

Thorctl also has the ability to CaRT and unCaRT local files. This is particularly helpful if you want to download a file
in a CaRTed format and then unCaRT it in a quarantined location later or CaRT files to store after
analysis is complete.

#### CaRTing Files

To CaRT a file, simply run:

```bash
thorctl cart <path-to-file>
```

You can also CaRT multiple files in one command:

```bash
thorctl cart <path-to-file1> <path-to-file2> <path-to-file3>
```

##### Specifying an Output Directory
CaRTing with Thorctl will create a directory called "carted" in your current directory containing the CaRTed files
with the `.cart` extension. To specify an output directory to save the CaRTed files to, use the `-o` or `--output` flag:

```bash
thorctl cart --output ./files/my-carted-files <path-to-file>
```

##### CaRTing In-Place
You can also CaRT the files in-place, replacing the original files with the new CaRTed files, by using the
`--in-place` flag:

```bash
thorctl cart --in-place <path-to-file>
```

##### CaRTing Directories
Giving the path of a directory to CaRT will recursively CaRT every file within the directory.

```bash
thorctl cart <path-to-dir>
```

Because CaRTed files will be saved together in one output folder, collisions can occur if files have the same name
within a directory structure. For example, let's say I have a directory called `my-dir` with the following structure:

```
my-dir
├── dir1
│   └── malware.exe
└── dir2
    └── malware.exe
```

Because Thorctl will recursively CaRT all files within `my-dir` and save them in one output directory, one
`malware.exe.cart` will overwrite the other. To avoid such collisions, you can either use the aforementioned
`--in-place` flag to CaRT the files in-place or use the `-D` or `--preserve-dir-structure` flag to output files in a
structure identical to the input directory. So CaRTing `my-dir` with the above structure using the
`--preserve-dir-structure` option would yield the output directory `carted`, having the following structure:

```
carted
└── my-dir
    ├── dir1
    │   └── malware.exe.cart
    └── dir2
        └── malware.exe.cart
```

##### Filtering Which Files to CaRT
There may be cases where you want to CaRT only certain files within a folder. Thorctl provides the ability to either
inclusively or exclusively filter with regular expressions using the `--filter` and `--skip` flags, respectively.
For example, to CaRT only files with the `.exe` extension within a directory, you could run the following command:

```bash
thorctl files cart --filter .*\.exe ./my-dir
```

Or to CaRT everything within a directory except for files starting with `temp-`, you could run this command:

```bash
thorctl files cart --skip temp-.* ./my-dir
```

Supply multiple filters by specifying filter flags multiple times:

```bash
thorctl files cart --filter .*\.exe --filter .*evil.* --skip temp-.* ./my-dir
```

The filter and skip regular expressions must adhere to the format used by the Rust
[regex crate](https://docs.rs/regex/latest/regex/#syntax). Fortunately, this format is very similar to
most other popular regex types and should be relatively familiar. A helpful site to build and test your
regular expressions can be found here: [https://rustexp.lpil.uk](https://rustexp.lpil.uk/)

#### UnCaRTing Files

UnCaRTing in Thorctl looks very similar to CaRTing as explained above but uses the `uncart` command instead:

```bash
thorctl uncart <path-to-CaRT-file>
```

You can specify multiple CaRT files, unCaRT in-place, preserve the input directory structure, and apply filename
filters just as with the `cart` command. For example:

```bash
thorctl uncart --filter .*\.cart --skip temp-.* --output ./my-output --preserve-dir-structure ./my-carts hello.cart
```
