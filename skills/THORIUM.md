---
name: thorium
description: Guide for AI agents to interact with Thorium, a highly scalable file analysis and data generation framework. Covers instance discovery, API routes, thorctl CLI, documentation, and workflow references.
---

# Thorium

Thorium is a highly scalable, distributed malware analysis and data generation framework built for cyber incident response, triage, and file analysis. It provides safe ingestion/storage of data, automated analysis in sandboxed environments, and easy access to results and metadata.

Source code: `github.com/cisagov/thorium`

## Step 1: Discover the Thorium Instance

Before doing anything else, determine the Thorium instance URL. Follow these steps in order:

1. **Check for a running local instance:**
   ```sh
   curl -sf -o /dev/null -w '%{http_code}' http://localhost:8080/api/health
   ```
   If the response is `204`, a local instance is running. Set `THORIUM_URL=http://localhost:8080`.

2. **If not reachable, check if minithor is available and try to expose:**
   ```sh
   test -x /usr/bin/minithor
   ```
   If minithor exists, run:
   ```sh
   /usr/bin/minithor expose
   ```
   Wait a few seconds, then re-check health:
   ```sh
   curl -sf -o /dev/null -w '%{http_code}' http://localhost:8080/api/health
   ```
   If `204`, set `THORIUM_URL=http://localhost:8080`.

3. **If still not reachable (or minithor is not installed):**
   Ask the user: "No local Thorium instance detected. Please provide the URL to your Thorium instance (e.g., `https://thorium.example.com`)."
   Set `THORIUM_URL` to the user-provided URL.

Use `THORIUM_URL` as the base for all URLs below.

## Step 2: Resource Directory

These are the key resources available on a Thorium instance:

| Resource | URL | Description |
|----------|-----|-------------|
| Web UI | `{THORIUM_URL}` | Browser interface for file browsing, uploading, reactions, search, and results |
| User Docs | `{THORIUM_URL}/api/docs/user/index.html` | Guides for uploading files, viewing results, spawning reactions, search, tagging. If the API is offline, use `https://cisagov.github.io/thorium/` as a fallback |
| Developer Docs | `{THORIUM_URL}/api/docs/dev/thorium/index.html` | Rust API documentation (rustdoc) for Thorium internals |
| API Spec (Swagger) | `{THORIUM_URL}/api/docs/swagger-ui/` | Interactive OpenAPI spec — full details on every endpoint, request/response schemas |
| MCP Server | `{THORIUM_URL}/api/mcp` | Model Context Protocol endpoint for AI tool use (requires Authorization header) |

## Step 3: API Authentication

All API routes (except `/api/health`, `/api/banner`, and `/api/`) require authentication via the `Authorization` header. Two methods are supported:

### Method 1: Username/Password (Basic Auth)

Authenticate with HTTP Basic auth to obtain a token. Base64-encode `username:password` and send:

```
Authorization: Basic <base64(username:password)>
```

Call `POST /api/users/auth` with this header to receive a JSON response containing a `token` and its `expires` datetime.

### Method 2: Token Auth (Preferred)

Once you have a token (from Basic auth, `thorctl login`, or the config file at `~/.thorium/config.yml`), base64-encode the token and send:

```
Authorization: Token <base64(token)>
```

or equivalently:

```
Authorization: Bearer <base64(token)>
```

Token auth is used for all subsequent API requests. Tokens expire and must be refreshed by re-authenticating.

### Using the Rust Client

The Thorium Rust client (`thorium::Thorium`) handles auth automatically:

```rust
// With username/password:
let client = Thorium::build("http://localhost:8080")
    .basic_auth("user", "password")
    .build().await?;

// With an existing token:
let client = Thorium::build("http://localhost:8080")
    .token("my-token")
    .build().await?;

// From a keys file (contains api, username/password or token):
let client = Thorium::build("http://localhost:8080")
    .from_keys("/path/to/keys.yml")?
    .build().await?;

// From a thorctl config (CtlConf):
let ctl_conf = thorium::CtlConf::new("~/.thorium/config.yml")?;
let client = Thorium::build(&ctl_conf.keys.api)
    .from_ctl_conf(ctl_conf)?
    .build().await?;
```

### Using curl

```sh
# Authenticate and get a token:
curl -X POST -u "username:password" {THORIUM_URL}/api/users/auth

# Use the token for subsequent requests (base64-encode the token):
curl -H "Authorization: Token $(echo -n 'YOUR_TOKEN' | base64)" {THORIUM_URL}/api/version
```

## Step 4: API Route Reference

All API routes are under `/api/`.

### Core Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/` | Identify this API as Thorium |
| `GET` | `/api/health` | Health check — `204` healthy, `503` unhealthy |
| `GET` | `/api/version` | Current Thorium version (requires auth) |
| `GET` | `/api/banner` | Login banner message |

### Domain Routes

See Swagger UI for complete endpoint details including request/response schemas.

| Area | Route Prefix | Key Operations |
|------|-------------|----------------|
| Files | `/api/files/` | Upload, download (CaRT or encrypted ZIP), list, count, update, delete, tag, comment, list associations, view/upload results |
| Groups | `/api/groups/` | Create, list, update, delete groups; LDAP sync; group stats |
| Images (Tools) | `/api/images/` | Create, list, update, delete analysis tools; manage notifications and bans |
| Pipelines | `/api/pipelines/` | Create, list, update, delete tool chains; manage notifications and bans |
| Reactions | `/api/reactions/` | Create, list, get, update, delete reactions; view/set logs; handle lifecycle (proceed/fail); manage cache and sub-reactions |
| Results | `/api/files/results/:sha256` | Get and upload tool results and result files for a sample |
| Repos | `/api/repos/` | Ingest, list, download, tag repos; list commitishes; view/upload results |
| Search | `/api/search/` | Full-text search across sample results and tags (defaults to samples; repo indexes also available) |
| Streams | `/api/streams/` | Query stream depth (object counts over time ranges) and map data |
| Users | `/api/users/` | Create, list, update, delete users; authenticate; email verification; LDAP sync |
| System | `/api/system/` | System init, settings, stats, backup/restore, cache reset, node/worker management |
| Trees | `/api/trees/` | Build and retrieve relationship trees from samples, repos, or entities |
| Entities | `/api/entities/` | Create, list, update, delete entities; manage entity tags and images |
| Associations | `/api/associations/` | Create associations between entities and other objects |
| Binaries | `/api/binaries/` | Download thorctl and other Thorium binaries |
| Events | `/api/events/` | Internal event queue operations (pop, clear, reset) |
| Jobs | `/api/jobs/` | Internal job lifecycle (claim, proceed, error, sleep, checkpoint, bulk reset) |
| Network Policies | `/api/network-policies/` | Create, list, update, delete network policies; view group defaults |

### MCP Tools

The MCP server at `/api/mcp` exposes these tools (requires `Authorization` header):

- `get_sample` — Get basic info about a specific sample/file by SHA256
- `get_sample_results` — Get tool results for a specific sample/file by SHA256
- `list_sample_result_file_paths` — List all available result files for a tool for a specific sample/file
- `get_sample_result_file` — Get a specific result file from a tool for a specific sample
- `list_images` — List the images (analysis tools) in a group
- `list_pipelines` — List the pipelines in a group
- `start_tree` — Find data related to a sample, repo, tag, or entity by crawling data in a tree structure

## Step 5: thorctl CLI (User/Developer Tool)

`thorctl` is the command-line tool for large-scale Thorium operations.

### Installation

`thorctl` comes pre-installed with `minithor deploy`. Check if it is already available:

```sh
which thorctl
```

If not found, install it manually:

```sh
curl {THORIUM_URL}/api/binaries/install-thorctl.sh | bash -s -- {THORIUM_URL}
```

### Authentication

```sh
thorctl login {THORIUM_URL}
```

Config is stored at `~/.thorium/config.yml`.

### Key Subcommands

| Command | Description |
|---------|-------------|
| `thorctl login <url>` | Authenticate with a Thorium instance |
| `thorctl clusters` | View cluster node status and worker info |
| `thorctl groups` | List and describe groups you belong to |
| `thorctl files` | Upload, download, get, count, describe, and delete files |
| `thorctl images` | Get, describe, edit, import/export images; manage image bans and notifications |
| `thorctl pipelines` | Get, describe, import/export pipelines; manage pipeline bans and notifications |
| `thorctl reactions` | Create, get, describe, delete reactions; retrieve reaction logs |
| `thorctl results` | Download results for files/repos and upload new results |
| `thorctl tags` | Get, add, and delete tags on files and repos |
| `thorctl repos` | Ingest, update, get, describe, download, compile repos; list commits and contributors |
| `thorctl network-policies` | Get, describe, create, update, delete, and verify network policies; view group defaults |
| `thorctl ai` | Chat with AI and summarize files/repos |
| `thorctl cart` | Cart (encrypt/neuter) files locally |
| `thorctl uncart` | Uncart files locally |
| `thorctl run` | Create a reaction, monitor it, and download results in one step |
| `thorctl toolbox` | Import and update toolboxes (pre-configured tool/pipeline collections) |
| `thorctl config` | Modify thorctl configuration settings |
| `thorctl update` | Update thorctl binary |

Use `thorctl -h` or `thorctl <subcommand> -h` for detailed help on any command.

## Step 6: thoradm CLI (Admin Tool)

`thoradm` is the admin-only CLI for Thorium infrastructure management. It requires both a thorctl config (for user auth) and a cluster config (`thorium.yml`) containing database credentials.

Check if it is available:

```sh
which thoradm
```

### Getting the Cluster Config

thoradm requires a `thorium.yml` cluster config file containing database credentials (Elastic, Redis, Scylla, etc.). There are two ways to obtain it:

**Option 1: minithor (local development)**

```sh
minithor get-config
```

This writes the config to `~/thorium.yml`.

**Option 2: kubectl (external Kubernetes cluster)**

Extract the config directly from the cluster secret:

```sh
kubectl get secret thorium -n thorium \
    -o go-template='{{index .data "thorium.yml" | base64decode}}' \
    > thorium.yml
```

By default, thoradm looks for `thorium.yml` in the current directory. Override with:

```sh
thoradm -c <PATH-TO-THORIUM.YML>
# or: thoradm --cluster-conf <PATH-TO-THORIUM.YML>
```

### Key Subcommands

| Command | Description |
|---------|-------------|
| `thoradm backup new` | Take a full backup of Thorium data (Redis, S3, tags, metadata) |
| `thoradm backup restore --backup <path>` | Restore a backup (overwrites all current data) |
| `thoradm backup scrub --backup <path>` | Verify backup integrity via checksums |
| `thoradm settings get` | View current system settings (resource limits, fairshare, host paths) |
| `thoradm settings update` | Update system settings (reserved CPU/memory/storage, fairshare, host path whitelist) |
| `thoradm settings reset` | Reset all system settings to defaults |
| `thoradm settings scan` | Run a manual consistency scan across all Thorium data |
| `thoradm provision node` | Provision a Kubernetes node for Thorium job scheduling |
| `thoradm census new` | Run a data consistency census (all, tags, files, repos, commitishes) |

Use `thoradm -h` or `thoradm <subcommand> -h` for detailed help on any command.

For full thoradm documentation, see the User Docs at `{THORIUM_URL}/api/docs/user/admins/thoradm/thoradm.html`.

## Step 7: minithor (Local Development)

Not all Thorium deployments use minithor — it is specific to local development environments running on Minikube. Check if it is available before using any of the commands below:

```sh
which minithor
```

If `minithor` is not found, this environment may be using a different deployment method and these commands will not apply.

| Command | Description |
|---------|-------------|
| `minithor minikube install` | Install minikube and start a Kubernetes cluster |
| `minithor deploy` | Deploy all Thorium services and backing infrastructure |
| `minithor expose` | Port-forward Thorium API to localhost:8080 |
| `minithor expose --port <port>` | Port-forward to a custom local port |
| `minithor expose --dev` | Also forward database ports (Elastic, Redis, MinIO, Scylla) |
| `minithor expose --status` | Show which port-forwards are running |
| `minithor expose --stop` | Stop all port-forwards |
| `minithor start` | Start a previously stopped cluster |
| `minithor stop` | Stop the cluster (preserves state) |
| `minithor get-config` | Extract running config to ~/thorium.yml |
| `minithor cleanup --confirm` | Remove all Thorium resources for a fresh deploy |

## Step 8: Python Client

The `thorpy` Python client library (in the source repo at `python/thorpy/`) is a synchronous Python wrapper around the Rust client built with PyO3. Type stubs are in the source repo at `python/thorpy/thorium.pyi` for IDE support.

Currently implemented features are limited to: health, identify, and reaction create/create_bulk. See the source repo at `python/thorpy/README.md` for building and usage details.

## Step 9: Key Concepts

### CaRT Format and File Downloads

All files uploaded to Thorium are stored in CaRT (encrypted/neutered) format to prevent accidental execution of malicious content. Files can be downloaded in two formats:

- **CaRT** (default): encrypted/neutered container format. Use `thorctl cart` / `thorctl uncart` to convert files locally, or download uncarted via `thorctl files download --uncarted`.
- **Encrypted ZIP**: available via the Web UI and the API at `GET /api/files/sample/:sha256/download/zip` (accepts an optional password parameter). This option is not currently available through thorctl.

### Groups

Groups are the primary access control mechanism. Files, images, pipelines, and reactions all belong to groups. Users must be members of a group to view or modify its resources.

### Images vs Pipelines vs Reactions

- **Images** are individual analysis tools (container-based) that run against files
- **Pipelines** are ordered chains of images that define a multi-step analysis workflow
- **Reactions** are instances of a pipeline running against one or more resources. A reaction can target:
  - **Samples** — files identified by SHA256
  - **Repos** — Git repositories (with optional branch/commit/tag)
  - **Buffers** — ephemeral data passed directly to the job
  - **None** — passing no resources creates a generic job (useful for data generation or maintenance tasks)
  
  Reactions track status, logs, and results for each stage in the pipeline.

### Entities, Associations, and Trees

**Entities** are user-created objects that represent real-world things in Thorium. Entity kinds include:
- **Device** — a physical or virtual device (e.g., a router, server, IoT device)
- **Vendor** — a manufacturer or software vendor
- **Collection** — a dynamic list of items in Thorium based on search parameters like tags
- **FileSystem** — a filesystem extracted or carved from a sample
- **Folder** — a folder within a filesystem
- **Other** — anything that doesn't fit the above categories

Entities support tags, images, and group-based access control.

**Associations** are directional (or bidirectional) relationships that link entities, files, and repos to each other. Association kinds include: FileFor, DocumentationFor, FirmwareFor, AssociatedWith, DevelopedBy, ContainsCVE, ContainsCWE, BasedIn, EmployedBy, ParentCompanyOf, UsedBy, UsedIn, PerformedBy, FileSystemIn, FolderIn, and FileIn.

**Trees** build relationship graphs across samples, repos, and entities to visualize how data is connected.

## Where to Go Next

- For **full API details**: open the Swagger UI at `{THORIUM_URL}/api/docs/swagger-ui/`
- For **user workflows** (uploading, searching, reactions): read the User Docs at `{THORIUM_URL}/api/docs/user/index.html`
- For **developing tools/pipelines**: see the Developer Docs at `{THORIUM_URL}/api/docs/dev/thorium/index.html` and the Workflow Skills section
- For **admin operations** (backup, settings, provisioning): see `thoradm -h` and the docs at `{THORIUM_URL}/api/docs/user/admins/thoradm/thoradm.html`
- For **source code and internals**: browse `github.com/cisagov/thorium` and the Developer Docs at `{THORIUM_URL}/api/docs/dev/thorium/index.html`
