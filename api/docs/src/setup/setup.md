# Setup

Before deploying Thorium ensure you have:
- A Kubernetes cluster
- A Redis instance
- A Scylla cluster/instance

Redis is used for all hot or time sensitive job data while Scylla stores all logs.
This is done to keep Thorium fast while minimize the memory used by Redis. Cassandra
may also be used in place of Scylla as the two expose compatible API's.

Deploying Thorium starts with configuring Thorium by editing thorium.yml and optionally
changing any settings in the deployment files within the inventory folder.

Here is an example thorium.yaml. The only settings you are required to change (assuming
you do not change the inventory files at all) are:

- thorium.secret_key (This should be kept secure)
- thorium.nodes (The nodes that Thorium can spawn pods on)
- redis.host (The IP/hostname the api can reach Redis at)
- scylla.nodes (A list of IPs/hostnames the api can reach Scylla at)

You can also enable LDAP settings if you wish to authenticate against LDAP. You can read
more about [LDAP usage with Thorium](../concepts/groups/groups.md).

```yaml
# Thorium settings
thorium:
  # The interface the Thorium api should bind to
  interface: "0.0.0.0"
  # The port Thorium should bind to
  port: 80
  # The namespace to use in the redis and the k8s namespace for Thorium system pods
  namespace: "thorium"
  # A secret key used for generating secrets and for bootstrapping with the deployer
  # Make sure this is a longer string and its protected
  secret_key: <CHANGE_ME>
  # The host/IP that the deployer can reach the API once its deployed at
  # This must start with http:// or https://
  external: "http://<127.0.0.1:30030>"
  # The log level of the API (and only the API currently)
  # The scaler determines log level by how it is compiled
  log_level: "info"
  # How long job data should be retained for after they are completed in seconds
  retention: 604800
  # Cross-Origin Resource Sharing settings
  cors:
    # Whether to allow CORS requests from any domain
    insecure: false
    # Specific domains to allow CORS requests from
    domains: []
  # The Authentication settings to use
  auth:
    # How long a users token can live for in days
    token_expire: 90
    # The settings to use for LDAP 
    # Uncommenting this LDAP section will disable redis backed basic auth for new users
    #ldap:
      # The hostname ldap can be reached at including "ldap://" or "ldaps://"
      #host: "ldaps://<LDAP_SERVER>"
      # The filters to append to uid=<username> when binding in ldap
      #bind_filters: "<BIND_FILTER>"
      # The filters to append to cn=<"group"> when searching in ldap
      #search_filters: "<SEARCH_FILTER>"
  # The settings used for the scaler
  scaler:
    # How long the cache should live for at most before being invalidated in seconds
    cache_lifetime: 600
    # The number of deadlines to pull for one scale loop
    deadline_window: 100000
    # The specific settings for kubernetes
    k8s:
      # The nodes that are in this k8s cluster and that Thorium can use
      #nodes:
        #- "<NODE1>"
        #- "<NODE2>"
        #- "<NODE3>"
        #- "<NODE4>"
      # What node you want to filter from running Thorium jobs
      filters:
        # whether to run jobs on master nodes or not
        master: false,
        # labels to require Thorium enabled nodes to have
        custom: []
      # The maximum number of pods to spawn in one scale loop
      max_sway: 50
      # How long we should sleep between each scale attempt
      dwell: 5
      # Whether privilege escalation is allowed in pods spawned in kubernetes
      privilege_escalation: false
# Redis specific settings
redis:
  # The host/IP that redis can be reached at
  host: "<REDIS_HOST>"
  # The port Redis is bound too
  port: 6379
  # How large the connection pool should be to Redis
  pool_size: 50
  # If Redis has authentication enabled then a username and or password combo can be set
  # A username is not required but if the username is set then a password must also be
  username: "<REDIS_USER>"
  # The password to access Redis
  password: "<PASSWORD>"
# Scylla specific settings
scylla:
  # The hosts/IPs to reach scylla at
  nodes:
    - "<SCYLLA_HOST_1>"
    - "<SCYLLA_HOST_2>"
    - "<SCYLLA_HOST_3>"
  # How many times to replicate data across nodes
  replication: 2
```

Once you have done that you can deploy Thorium by running the deployer with the
following command. If you're working with a private registry you must also pass in
--private-registry. This will tell the deployer to also create a registry-token in
Thorium's namespace using the current users ~/.docker/config.json.

```
# if thorium.yml is in the current working directory
./deployer --cmd install
# if thorium.yml is in the current working directory 
# and your working with a private docker registry
./deployer --cmd install --private-registry
# if thorium.yml is not in the current working directory
./deployer --cmd install --config <path/to/thorium.yaml>
```

Make sure that the deployer is not being routed through your proxy if k8s/services
will not be reachable through it.

The deployer will create a Thorium service account with a randomly generated
256 bit long password. This account should not be used by anything other then
the Thorium system in normal circumstances.

The deployer will then ask you for the username/password you want it to deploy
as your admin account. You can then use this account to create more admins later.
If you get an error due to the password not being the same during this step you
do not have to completely reinstall Thorium. Just create the admin with the
command below.

You can always create more admins with
```
# if thorium.yml is in the current working directory
./deployer --cmd add_admin
# if thorium.yml is not in the current working directory
./deployer --cmd add_admin --config <path/to/thorium.yaml>
```

You now have a functioning Thorium deployment and can start running jobs.
