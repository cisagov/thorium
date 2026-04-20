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
   command -v minithor || test -x /usr/bin/minithor
   ```
   `minithor` is normally on your `PATH`; some environments instead place it at
   `/usr/bin/minithor`, so fall back to that path if it isn't found. If minithor exists, run:
   ```sh
   minithor expose
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

> **Sandboxed agents (e.g. Claude Code):** If you are a coding agent running commands in a network
> sandbox, connections to the Thorium API — including a local minithor at `http://localhost:8080`
> and any `minithor expose` port-forward — may be blocked by default. If health checks, `thorctl`,
> `curl`, or API calls fail with connection-refused/permission/"operation not permitted" errors that
> don't match a real outage, the request likely needs to reach Thorium **outside the sandbox**:
> allowlist the Thorium host/port (e.g. `localhost:8080`, the backing-service ports from
> `minithor expose --dev`, and any remote `THORIUM_URL`) or run that command with the sandbox
> disabled. Use your agent's sandbox controls to permit the connection before assuming the instance
> is down.

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

The Thorium API is a REST API served under `{THORIUM_URL}/api/`. Connecting to it is the same for
a **local** instance (e.g. minithor at `http://localhost:8080`, see Step 1) and a **remote** one —
only the base URL differs. To explore every endpoint and its request/response schema interactively,
open the Swagger UI at `{THORIUM_URL}/api/docs/swagger-ui/`.

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

`thorctl` is the command-line tool for large-scale Thorium operations (file/repo upload and
download, image/pipeline/toolbox management, running reactions, tagging, search, and more). It
talks to the same REST API as the Web UI and works against any reachable instance — local
(minithor) or remote.

### Setup (get started)

1. **Obtain the binary.** `minithor deploy` installs `thorctl` automatically. Otherwise check for
   it and install from the instance if missing:

   ```sh
   which thorctl  # already installed?
   # install (or update) directly from the target instance:
   curl {THORIUM_URL}/api/binaries/install-thorctl.sh | bash -s -- {THORIUM_URL}
   ```

   The script downloads the build matching the instance; `thorctl update` updates it later.

2. **Authenticate / choose a cluster.** `login` points thorctl at an instance and stores a token:

   ```sh
   thorctl login {THORIUM_URL}                 # prompts for username + password interactively
   thorctl login {THORIUM_URL} --user <name>   # non-interactive: add --password <pass> for CI
   thorctl login {THORIUM_URL} --clear-settings # reset config and start fresh for this instance
   ```

   For self-signed/dev certs, `login` accepts `--invalid-certs`, `--invalid-hostnames`, and
   `--certificate-authorites <path>`.

3. **Config file.** Credentials and settings are written to `~/.thorium/config.yml`. Point any
   command at a different config (e.g. a second cluster) with the global `--config <path>` flag, or
   supply a standalone keys file with `--keys <path>`. `thorctl config` edits these settings.

See `{THORIUM_URL}/api/docs/user/getting_started/thorctl.html` for per-OS install detail, and
`{THORIUM_URL}/api/docs/user/getting_started/login.html` /
`{THORIUM_URL}/api/docs/user/getting_started/registration.html` for account/login detail.

### Key Subcommands

| Command | Description |
|---------|-------------|
| `thorctl login <url>` | Authenticate with a Thorium instance (see Setup above) |
| `thorctl clusters status` / `workers` | View cluster node resource status and per-worker info |
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
| `thorctl toolbox` | Scaffold (`init`), `build`, `build-images`, `import`, `export`, `diff`, and `remove` toolboxes — portable tool/pipeline collections (see Step 10 and `{THORIUM_URL}/api/docs/user/developers/toolbox.html`) |
| `thorctl config` | Modify thorctl configuration settings |
| `thorctl update` | Update thorctl binary |

Use `thorctl -h` or `thorctl <subcommand> -h` for detailed help on any command.

> Building, integrating, or distributing your own tools? See **Step 10** for the developer
> workflow (image/pipeline `import`/`export` vs the `toolbox` subcommands, with tradeoffs).

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

> The raw config points at in-cluster `*.svc.cluster.local` addresses. If you are running
> `thoradm` (or any Thorium binary) **from your host** against a minithor cluster, the
> databases are only reachable through `minithor expose --dev`; use
> `minithor get-config --local` to get a config rewritten for those local ports. See
> "Step 7 → Testing Locally" for details (Scylla in particular needs special handling).

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

`minithor` stands up a **single-node Thorium stack on Minikube** — a self-contained local instance
for developing features against, and for building/testing/curating tools. It is not highly available
and is for development/testing only (or small offline fly-away kits), not production. A deploy brings
up all backing services (Redis, Elasticsearch, ScyllaDB, MinIO/S3, Postgres, Quickwit, Jaeger), the
Thorium operator and `ThoriumCluster`, a default admin user, a `static` group, an `allow-all` network
policy, and (optionally) an in-cluster container registry; it also installs `thorctl` and imports the
default toolbox. Backing-service passwords are randomly generated per deploy.

Requirements: a container runtime (podman or docker; `kvm2` also works on Linux) and a beefy host
(16+ GiB RAM, 8+ CPUs, 100+ GiB disk). It picks the best available driver (podman > docker > kvm2).

Not all Thorium deployments use minithor. Check if it is available before using any of the commands below:

```sh
command -v minithor || test -x /usr/bin/minithor
```

`minithor` is normally on your `PATH`; some environments instead place it at `/usr/bin/minithor`,
so check that path if it isn't found. If neither resolves, this environment uses a different
deployment method and these commands do not apply.

### Stand it up

```sh
minithor minikube install   # first time only: install minikube + start the k8s cluster
minithor deploy             # deploy all services + create admin user (default: test / INSECURE_DEV_PASSWORD)
minithor expose             # port-forward the API to http://localhost:8080
```

`deploy` accepts `--user`/`--password`/`--rand-password` to set the admin credentials,
`--registry`/`--registry-user <name>` to deploy the container registry, and `--config <path>` for a
custom `thorium-cluster.yml`. After `minithor stop` or a reboot, resume with `minithor start` then
`minithor expose` again.

### Reach the running instance

Once exposed, `THORIUM_URL=http://localhost:8080` (Step 1). Log in via the UI or
`thorctl login http://localhost:8080` with the deploy credentials, or hit the API directly (Step 3).
The instance serves its own docs at the same paths under `{THORIUM_URL}/api/docs/...` (Step 2) —
the user docs, Swagger UI, and rustdoc are all reachable locally at `http://localhost:8080/api/docs/...`,
so every doc link in this guide resolves against the running minithor instance, not just a remote one.
`minithor expose --dev` also forwards the backing-service ports (Elastic, Kibana, Redis, MinIO,
Scylla), and `minithor get-config` writes the cluster config (DB credentials) to `~/thorium.yml` —
needed by `thoradm` (Step 6).

| Command | Description |
|---------|-------------|
| `minithor minikube install` | Install minikube and start a Kubernetes cluster |
| `minithor deploy` | Deploy all Thorium services and backing infrastructure (add `--registry` for an in-cluster registry) |
| `minithor expose` | Port-forward the Thorium API to localhost:8080 (and the registry to localhost:5000 when one was deployed) |
| `minithor expose --port <port>` | Port-forward the API to a custom local port |
| `minithor expose --dev` | Also forward database ports (Elastic, Kibana, Redis, MinIO, Scylla). See "Testing locally" below — Scylla requires special handling |
| `minithor expose --status` | Show which port-forwards are running |
| `minithor expose --stop` | Stop all port-forwards **and remove any loopback aliases** created for Scylla |
| `minithor start` | Start a previously stopped cluster |
| `minithor stop` | Stop the cluster (preserves state) |
| `minithor get-config` | Extract the raw in-cluster config to ~/thorium.yml |
| `minithor get-config --local` | Extract the config **rewritten for local host access** via the exposed ports, to ~/thorium.local.yml (see "Testing locally") |
| `minithor cleanup --confirm` | Remove all Thorium resources for a fresh deploy (also stops port-forwards and removes loopback aliases) |
| `minithor minikube delete --confirm` | Fully remove minikube (also stops port-forwards and removes loopback aliases) |

### Testing Locally: Connecting a Local Build to the Cluster Databases

To run a local build of the API (or other Thorium services/tools) against the databases
running in the minikube cluster, you need two things: the database ports forwarded to your
host, and a config whose connection fields point at those local ports.

**1. Forward the database ports:**

```sh
minithor expose --dev
```

This forwards, in addition to the API (`localhost:8080`) and the registry (`localhost:5000`,
when one was deployed):

| Service | Local address | Notes |
|---------|---------------|-------|
| Elasticsearch | `https://localhost:9200` | self-signed cert — clients must skip cert validation (see below) |
| Kibana | `https://localhost:5601` | |
| Redis | `localhost:6379` | |
| MinIO (S3) | `http://localhost:9000` | |
| Scylla (CQL) | `<advertised-cluster-ip>:9042` | **not** `localhost` — see "How Scylla is exposed" |

**2. Generate a host-ready config:**

```sh
minithor get-config --local      # writes ~/thorium.local.yml
```

`--local` rewrites the in-cluster config so a host process can connect through the exposed
ports. It changes: Elastic/Redis/MinIO hosts → `localhost`; Elastic `insecure_certificates`
→ `true` (the cluster cert's SAN does not cover `localhost`); `scylla.nodes` → each node's
advertised cluster IP (matching the forward, see below); `thorium.port` → `8888` (binding
`80` needs root); and `thorium.tracing.external` → `null` (the in-cluster collector is not
reachable and only produces connection noise). The raw `~/thorium.yml` is left untouched.

> Plain `minithor get-config` (no `--local`) dumps the **raw** in-cluster config, whose
> hosts are `*.svc.cluster.local` DNS names that do **not** resolve from your host. Use it
> only for tools that run inside the cluster network; use `--local` for host testing.

**3. Run the local build:**

```sh
# from the repo root (so banner.txt / UI assets resolve)
cargo build -p thorium-api --bin thorium-api          # default features include `api`
./target/debug/thorium-api --config ~/thorium.local.yml
# the API connects to redis → scylla → elastic → s3 at startup, then:
curl -sf -o /dev/null -w '%{http_code}\n' http://localhost:8888/api/health   # expect 204
```

A `204` from `/api/health` means **all** databases connected — the API establishes its
Redis, Scylla, Elastic, and S3 connections eagerly during startup and will not bind/serve
until they all succeed.

### How Scylla is exposed (and why it's special)

Redis, Elasticsearch, and MinIO are single-endpoint services, so a normal
`localhost:<port>` port-forward works. **Scylla is different**: the CQL driver connects to
the seed node, then reads the cluster topology and *re-dials each node at the address it
advertises* (`broadcast_rpc_address`). With the scylla-operator each node advertises its
per-pod ClusterIP (e.g. `10.98.161.114`), which a `localhost` tunnel cannot satisfy — so a
naive forward connects, authenticates, then times out setting up the connection pool.

To make this work with only a port-forward, `minithor expose --dev`:

1. discovers each Scylla node's advertised cluster IP (its per-pod member service ClusterIP),
2. **adds that IP as a loopback alias** on your machine (`ip addr add <ip>/32 dev lo` on
   Linux, `ifconfig lo0 alias <ip>` on macOS) so the host can bind it, and
3. binds the forward to that exact IP (`kubectl port-forward --address <ip> … 9042:9042`).

The seed connection and the driver's topology re-dial then both land on the forward.
`get-config --local` writes that same advertised IP into `scylla.nodes`. The created
aliases are tracked in `/tmp/minithor-expose-logs/loopback-aliases` and are removed
automatically by `expose --stop`, `cleanup --confirm`, and `minikube delete --confirm`.

> Adding a loopback alias requires `sudo` (you may be prompted). The alias is harmless and
> host-local; if a forward is killed without `expose --stop`, the alias may linger until the
> next `expose --stop`/`cleanup` or a reboot.

### Troubleshooting Local Connectivity

First confirm what is actually forwarded and which aliases exist:

```sh
minithor expose --status                 # lists running port-forwards
ip addr show dev lo | grep -w inet       # Linux: shows loopback aliases (e.g. 10.x for Scylla)
ifconfig lo0 | grep inet                 # macOS equivalent
cat /tmp/minithor-expose-logs/*.log      # per-forward kubectl logs
```

| Symptom | Likely cause / fix |
|---------|--------------------|
| API panics: `Failed to setup keyspace: … ConnectTimeout` or hangs after "Authenticating to Scylla" | Scylla reached but its advertised address isn't reachable. Run `minithor expose --dev` (not a plain forward) and point `scylla.nodes` at the advertised IP — easiest via `get-config --local`. Verify the loopback alias exists with `ip addr show dev lo`. |
| `expose --dev` prints `scylla: SKIPPED (no Scylla nodes found)` | The Scylla pods aren't up/ready yet, or are in a different namespace. Check `minikube kubectl -- get pods -n scylla`. |
| Elastic errors: TLS/cert hostname mismatch or `invalid peer certificate` | The cluster cert isn't valid for `localhost`. Set `elastic.insecure_certificates: true` (done automatically by `get-config --local`). |
| Elastic `403 security_exception` for `cluster:monitor/*` | Not a connection problem — the `thorium` user simply lacks that privilege. Auth succeeded; querying a Thorium index works. |
| API fails to bind / "permission denied" on port 80 | Running as non-root with `thorium.port: 80`. Use a non-privileged port (`get-config --local` sets `8888`). |
| `expose` reports a forward, but connections are refused | The forward process may have died; check its log under `/tmp/minithor-expose-logs/` and re-run `minithor expose --dev`. Stale forwards on a port are auto-killed on the next `expose`. |
| Stale Scylla loopback alias after a crash | `minithor expose --stop` removes all aliases; or remove manually with `sudo ip addr del <ip>/32 dev lo` (Linux) / `sudo ifconfig lo0 -alias <ip>` (macOS). |

**macOS notes:** the loopback-alias and port-forward approach works on macOS too (it's
host-local and driver-agnostic — `kubectl port-forward` tunnels through the API server, so
no routing into the docker-driver VM is needed). minithor uses `ifconfig lo0 alias/-alias`
on macOS and falls back from `ss` to `lsof` for stale-port detection.

### Container Registry

`minithor deploy --registry` deploys a container registry in the `thorium` namespace; use
`--registry-user <name>` to enable it with basic auth (password auto-generated and printed).

| Context | Registry Address |
|---------|-----------------|
| From the host (push/pull) | `localhost:5000` (requires `minithor expose`) |
| From within Thorium (image references) | `registry.thorium.svc.cluster.local:5000` |

To push an image and reference it in Thorium:

```sh
# Tag and push from the host
docker tag myimage:latest localhost:5000/path/to/image:tag
docker push localhost:5000/path/to/image:tag
```

When configuring a Thorium image/tool to use it, reference the in-cluster address:

```
registry.thorium.svc.cluster.local:5000/path/to/image:tag
```

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

## Step 10: Building, Integrating & Testing Tools (Developer Workflow)

This section is for developers **building, integrating, and testing their own tools (images) and pipelines** in Thorium — packaging them as Thorium images, wiring them into pipelines, and distributing them as reusable toolboxes. It builds on the container registry (Step 7) and the images-vs-pipelines model (Step 9); see those rather than re-reading them here.

> **Full toolbox reference (deep links).** The authoritative toolbox docs are the *Toolboxes*
> page of the User Docs. Jump straight to a section (anchors follow the headings):
> - Overview & command table — `{THORIUM_URL}/api/docs/user/developers/toolbox.html#what-is-a-toolbox`
> - Which command/flag for which task — `…/toolbox.html#which-commandflag-for-which-task`
> - Toolbox layout — `…/toolbox.html#toolbox-layout`
> - Tool docs (`description.md`) — `…/toolbox.html#tool-docs-descriptionmd`
> - Toolbox config (`config.toml`) — `…/toolbox.html#toolbox-config-configtoml`
> - Image manifest (`manifest.toml`) — `…/toolbox.html#image-manifest-manifesttoml`
> - Reusing another image's container (`image_from`) — `…/toolbox.html#reusing-another-images-container-image_from`
> - Pipeline manifest (`manifest.toml`) — `…/toolbox.html#pipeline-manifest-manifesttoml`
> - How container image tags are built — `…/toolbox.html#how-container-image-tags-are-built`
> - Creating a toolbox from scratch — `…/toolbox.html#creating-a-toolbox-from-scratch`
> - Building images locally — `…/toolbox.html#building-images-locally`
> - Base images (`[base_image]`) — `…/toolbox.html#base-images-base_image`
> - Exporting from a running instance — `…/toolbox.html#exporting-from-a-running-instance`
> - Importing into an instance — `…/toolbox.html#importing-into-an-instance`
> - Diffing a toolbox against an instance — `…/toolbox.html#diffing-a-toolbox-against-an-instance`
> - Network policies — `…/toolbox.html#network-policies`
> - Removing a toolbox — `…/toolbox.html#removing-a-toolbox`
> - Offline transfer with bundled images — `…/toolbox.html#offline-transfer-with-bundled-images`
>
> (If the API is offline, the same page is mirrored under `https://cisagov.github.io/thorium/`.)

### When to use which

- **`thorctl images` / `thorctl pipelines` `import`/`export`** — round-trip a **single** image or pipeline that is **already integrated** into Thorium. Pull one existing tool's config (and its container image) out to disk, move it, and push it back. These operate on what already exists; they do **not** scaffold a new tool's config.
- **`thorctl toolbox` (`init` + `build` + `build-images` + `import`/`export`)** — **integrate brand-new tools** (toolbox `init` is the only command that *scaffolds* image/pipeline config files from scratch) and **curate, version, and distribute larger collections** of tools as a single unit (`toolbox.json`).
- **Both support offline bundling** for moving tools between segmented/air-gapped networks (via different mechanisms — see Tradeoffs).

### Working inside a toolbox — the build/test loop

A toolbox is a directory of tool folders rolled into one `toolbox.json`. It is **anchored on its
`config.toml`**: that file defines the toolbox identity (name, registry) and is what `build`,
`export`, and `init` resolve against.

```
my-toolbox/
├── config.toml              # toolbox-wide: name, registry, registries, image_path_prefix,
│                            #   export_image_path, export_pipeline_path, bundled_images, [base_image]
├── toolbox.json             # generated by `build` (and `export`)
├── images/<tool>/           # per image: manifest.toml + <name>.json + description.md + Dockerfile
│                            #   (+ <name>.tar.gz only in --with-images bundles)
└── pipelines/<tool>/        # per pipeline: manifest.toml + <name>.json + description.md
```

`build` discovers `manifest.toml` files at **any depth**, so the layout is flexible;
`export_image_path` / `export_pipeline_path` only control where `export` *places* dirs (default
`images`/`pipelines`).

**End-to-end loop to build out and test a new tool:**

```sh
# 1. Scaffold a toolbox: one or more image build dirs + a pipeline binding images.
#    config.toml lands in --toolbox-dir (default: cwd). Add -n for non-interactive defaults.
thorctl toolbox init toolbox -i images/clamav,images/yara -p pipelines/av:clamav,yara -g <group> \
    [--registry <reg>] [--image-path images --pipeline-path pipelines]

# 2. Edit the tool: write its Dockerfile and tune images/<tool>/<name>.json + manifest.toml
#    (description.md becomes the tool's description on build).

# 3. Build the manifest. --path (crawl root) and --output (toolbox.json) both DEFAULT to the
#    config.toml's directory; override independently. -c points at a non-cwd config.toml.
thorctl toolbox build                       # reads ./config.toml, writes ./toolbox.json
thorctl toolbox build -c sub/config.toml    # builds the toolbox in sub/, writes sub/toolbox.json

# 4. Build & push the container images straight from toolbox.json (forks without CI).
thorctl toolbox build-images ./toolbox.json --push      # everything buildable
thorctl toolbox build-images ./toolbox.json -i clamav   # just one image
#   useful: --no-cache, --pull, --tag-suffix -mybranch, --base-image ARG=IMAGE

# 5. Import into an instance to test (local path, directory, or URL).
thorctl toolbox import ./toolbox.json [--group-override <group>]

# 6. Run it (Step 9), inspect results, iterate, then re-build + re-import.
thorctl run ...   # or thorctl reactions create
```

**Appending into an existing toolbox** (add more tools in later runs — fully interoperable
because `build` re-walks the whole tree):

```sh
# add one image stub; --image-name overrides the registry tag leaf (default: dir name)
thorctl toolbox init image images/peid -g <group> [--image-name path/peid] [-c config.toml]
# add one pipeline stub; --order references images that must be in --images
thorctl toolbox init pipeline pipelines/triage -i clamav,yara -g <group> [-c config.toml]
thorctl toolbox build   # re-roll toolbox.json with the new tools
```

`init image`/`init pipeline` take an optional **`-c <config.toml>`** as a *resolution/validation*
source (NOT placement — the positional path is always the destination). With `-c`, `init pipeline`
requires every referenced image to already exist in that toolbox (it never creates image stubs) and
version-pins each from the toolbox; `init image -c` errors on a duplicate name+version unless
`--overwrite`. `init pipeline` also errors if `--order` names an image absent from `--images`, and
requires `--images` in `-n` mode.

**Tag derivation:** buildable images derive `<registry>/[<image_path_prefix>/]<name>:<version>`.
`thorctl toolbox build --use-image-path` swaps the tool `name` leaf for the manifest's path-style
`image_name`. (There is no `--flatten-image-paths` flag.)

### Distribute, diff, and apply a toolbox

```sh
thorctl toolbox export -g <group> -o ./tb [--with-images]    # pull a collection from an instance
thorctl toolbox export -p <grp>/<pipe> -i <grp>/<img> -o ./tb # specific pipelines + images
thorctl toolbox import ./tb/toolbox.json \                    # or a directory / URL
    [--group-override <group>] [--image-path-prefix <reg/base>] [--overwrite|--skip-conflicts] \
    [--rollback-on-failure] [--update-network-policy]
thorctl toolbox diff ./tb [--group-override <g>] [--exit-code]  # preview an import (CI drift gate)
thorctl toolbox remove ./tb [--group-override <g>]             # delete a toolbox's pipelines/images
```

**`export` is append-safe.** Exporting into a directory that already has a `config.toml` **reuses
and preserves it** (no `--config` needed); `--overwrite-config` replaces it. Settings-source
priority: `--config` (seed a *new* toolbox from another's settings) > existing `<output>/config.toml`
> `--name`/`--registry`. A `-p`/`-i` entry can carry a **`=dest`** suffix
(`-i group/name=tools/clamav`) to place that resource's files in a chosen subdir — placement only,
never affecting selection/membership/order; auto-pulled pipeline images use the default layout unless
named with their own `=dest`.

**Overwrite scopes are distinct flags:** `--overwrite` = per-tool files (and, in `import`, the
resource conflict mode — apply incoming); `--overwrite-config` = `config.toml` (export + init
toolbox); `--update-network-policy` = cluster network policies (import). `import`'s conflict modes
are interactive editor (default), `--overwrite`, or `--skip-conflicts` (skip differing resources,
the CI/agent-safe choice); `--rollback-on-failure` undoes a partial apply in non-interactive runs.

**Offline / air-gapped transfer:** `export --with-images` bundles each image's tarball into its
tool dir (recorded in `toolbox.json` so import finds it); `import --image-path-prefix <reg>`
loads, retags, and pushes them into the offline registry. You don't need to know the offline
registry at export time.

For registry addresses (e.g. `localhost:5000` vs `registry.thorium.svc.cluster.local:5000`), see Step 7.

### Single tools as code — `images`/`pipelines import`/`export`

For a one-off (a single already-integrated tool, no toolbox overhead):

```sh
thorctl images export -g <group> -o ./exports                 # all images in the group
thorctl images export -g <group> -o ./exports --config-only   # configs only, no docker images
thorctl pipelines export -g <group> -o ./exports
thorctl images import -g <group> -i ./exports \
    --registry localhost:5000 [--registry-override <url>] [--skip-push]
thorctl pipelines import -g <group> -i ./exports
```

`import` opens an interactive editor to resolve differences by default; use `--overwrite` (apply
incoming) or `--skip-conflicts` (skip differing resources, CI-safe), and `--rollback-on-failure` to
undo a partial apply. `--migrate-registry` only updates the registry path.

### Tradeoffs

- **Scaffolding:** only `toolbox init` creates a *new* tool's config; `images`/`pipelines export` can only extract something already in Thorium (and `import` needs a hand-written config). New tool → toolbox; existing tool → image/pipeline.
- **Scope & structure:** image/pipeline import/export is a flat, low-overhead per-resource round-trip; toolbox adds `config.toml`/`manifest.toml` and a build step — more setup, but you get shared registry config, collection-level versioning, `build-images`, `diff`, `--group-override`, and a single distributable `toolbox.json`.
- **Offline bundling (both, opposite defaults):** `images export` bundles the container image **by default** (opt out with `--config-only`), and `import` repoints it via `--registry`/`--registry-override`. `toolbox export` is **config-only by default** and bundles image tarballs only with `--with-images`, which `toolbox import` then pushes using `--image-path-prefix`. Mind this default-polarity difference when moving tools across networks.

### Typical dev loop

1. Write your tool and its Dockerfile.
2. Scaffold its Thorium config — `toolbox init image` (or `init pipeline`) for a new tool.
3. `toolbox build` → `toolbox.json`, then `toolbox build-images --push` to build/push its container (or build/push manually per Step 7).
4. Apply it — `toolbox import`, or `images`/`pipelines import` for a one-off.
5. Run it — `thorctl run` or `thorctl reactions create` against a sample/repo.
6. Iterate — edit configs, re-`build`, re-`import` (or `thorctl images edit` for a quick tweak).

## Where to Go Next

- For **full API details**: open the Swagger UI at `{THORIUM_URL}/api/docs/swagger-ui/`
- For **user workflows** (uploading, searching, reactions): read the User Docs at `{THORIUM_URL}/api/docs/user/index.html`
- For **developing tools/pipelines**: see the Developer Docs at `{THORIUM_URL}/api/docs/dev/thorium/index.html` and the Workflow Skills section
- For **packaging/distributing tools as toolboxes**: see Step 10 and the *Toolboxes* page at `{THORIUM_URL}/api/docs/user/developers/toolbox.html`
- For **admin operations** (backup, settings, provisioning): see `thoradm -h` and the docs at `{THORIUM_URL}/api/docs/user/admins/thoradm/thoradm.html`
- For **source code and internals**: browse `github.com/cisagov/thorium` and the Developer Docs at `{THORIUM_URL}/api/docs/dev/thorium/index.html`