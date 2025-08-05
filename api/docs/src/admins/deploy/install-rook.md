## Deploy Rook

This section will describe how to deploy a Rook Ceph cluster on K8s. This deployment will assume
the K8s cluster member nodes have attached unprovisioned raw storage devices. If you want to use host
storage from an existing mounted filesystem, review the
[rook docs](https://rook.github.io/docs/rook/latest/CRDs/Cluster/host-cluster/) before proceeding. 

> For single server Thorium deployments its best to skip deploying rook and just use a host path
> storageClass provisioner and Minio for better performance.

### 1) Create Rook CRD:

Apply the rook CRD and common resources.

```bash
kubectl apply -f https://raw.githubusercontent.com/rook/rook/refs/tags/v1.16.4/deploy/examples/crds.yaml
kubectl apply -f https://raw.githubusercontent.com/rook/rook/refs/tags/v1.16.4/deploy/examples/common.yaml
```

### 2) Create the Rook operator

You can deploy Rook Ceph with the default operator options. However, you may choose to disable certain drivers
such as CephFS that won't be needed for Thorium. To do that download the operator YAML resource definition and
modify it before applying it.

```bash
kubectl apply -f https://github.com/rook/rook/refs/tags/v1.16.4/deploy/examples/operator.yaml
```

### 3) Create Ceph/S3 Object Store

Create the Ceph pools and RADOS Object Gateway (RGW) instance(s). You may want to modify the redundancy
factors and number of gateway instances depending on the size of your K8s cluster. Some fields you may
look to modify are:

> The totals of `dataChunks` + `codingChunks` and separately `size` must both be <= the number of k8s
> cluster servers with attached storage that Rook can utilize. If this condition is not met, the Ceph
> cluster Rook deploys will not be in a healthy state after deployment and the Rook operator may fail
> to complete the deployment process.

- `spec.metadataPool.replicated.size` - Set to less than 3 for small k8s clusters
- `spec.dataPool.erasureCoded.dataChunks` - More erasure coding data chunks for better storage efficiency, but lower write performance
- `spec.dataPool.erasureCoded.codingChunks` - More erasure coding chunks for extra data redundancy
- `spec.gateway.instances` - Increase number of RGW pods for larger K8s clusters and better performance

```yaml,editable
cat <<EOF | kubectl apply -f -
#################################################################################################################
# Create an object store with settings for erasure coding for the data pool. A minimum of 3 nodes with OSDs are
# required in this example since failureDomain is host.
#  kubectl create -f object-ec.yaml
#################################################################################################################

apiVersion: ceph.rook.io/v1
kind: CephObjectStore
metadata:
  name: thorium-s3-store
  namespace: rook-ceph # namespace:cluster
spec:
  # The pool spec used to create the metadata pools. Must use replication.
  metadataPool:
    failureDomain: osd # host
    replicated:
      size: 3
      # Disallow setting pool with replica 1, this could lead to data loss without recovery.
      # Make sure you're *ABSOLUTELY CERTAIN* that is what you want
      requireSafeReplicaSize: true
    parameters:
      # Inline compression mode for the data pool
      # Further reference: https://docs.ceph.com/docs/master/rados/configuration/bluestore-config-ref/#inline-compression
      compression_mode: none
      # gives a hint (%) to Ceph in terms of expected consumption of the total cluster capacity of a given pool
      # for more info: https://docs.ceph.com/docs/master/rados/operations/placement-groups/#specifying-expected-pool-size
      #target_size_ratio: ".5"
  # The pool spec used to create the data pool. Can use replication or erasure coding.
  dataPool:
    failureDomain: osd # host
    erasureCoded:
      dataChunks:  3
      codingChunks:  2
    parameters:
      # Inline compression mode for the data pool
      # Further reference: https://docs.ceph.com/docs/master/rados/configuration/bluestore-config-ref/#inline-compression
      compression_mode: none
      # gives a hint (%) to Ceph in terms of expected consumption of the total cluster capacity of a given pool
      # for more info: https://docs.ceph.com/docs/master/rados/operations/placement-groups/#specifying-expected-pool-size
      #target_size_ratio: ".5"
  # Whether to preserve metadata and data pools on object store deletion
  preservePoolsOnDelete: true
  # The gateway service configuration
  gateway:
    # A reference to the secret in the rook namespace where the ssl certificate is stored
    sslCertificateRef:
    # The port that RGW pods will listen on (http)
    port: 80
    # The port that RGW pods will listen on (https). An ssl certificate is required.
    # securePort: 443
    # The number of pods in the rgw deployment
    instances: 1 # 3
    # The affinity rules to apply to the rgw deployment or daemonset.
    placement:
    #  nodeAffinity:
    #    requiredDuringSchedulingIgnoredDuringExecution:
    #      nodeSelectorTerms:
    #      - matchExpressions:
    #        - key: role
    #          operator: In
    #          values:
    #          - rgw-node
    #  tolerations:
    #  - key: rgw-node
    #    operator: Exists
    #  podAffinity:
    #  podAntiAffinity:
    # A key/value list of annotations
    annotations:
    #  key: value
    # A key/value list of labels
    labels:
    #  key: value
    resources:
    # The requests and limits set here, allow the object store gateway Pod(s) to use half of one CPU core and 1 gigabyte of memory
    #  limits:
    #    cpu: "500m"
    #    memory: "1024Mi"
    #  requests:
    #    cpu: "500m"
    #    memory: "1024Mi"
    # priorityClassName: my-priority-class
  #zone:
  #name: zone-a
  # service endpoint health check
  healthCheck:
    # Configure the pod probes for the rgw daemon
    startupProbe:
      disabled: false
    readinessProbe:
      disabled: false
EOF
```

### 4) Create block storage class

Use the following storage class to create a Rook Ceph data pool to store RADOS block devices (RBDs)
that will map to Kubernetes persistent volumes. The following command will create a block device pool
and storageClass (called `rook-ceph-block`). You will use this storage class name for creating PVCs
in the sections that follow. You may want to update the replication factors depending on the size
of your k8s cluster.

- `spec.replicated.size` - Set to less than 3 for small k8s clusters
- `spec.erasureCoded.dataChunks` - More erasure coding data chunks for better storage efficiency, but lower write performance
- `spec.erasureCoded.codingChunks` - More erasure coding chunks for extra data redundancy

```yaml,editable
cat <<EOF | kubectl apply -f -
#################################################################################################################
# Create a storage class with a data pool that uses erasure coding for a production environment.
# A metadata pool is created with replication enabled. A minimum of 3 nodes with OSDs are required in this
# example since the default failureDomain is host.
#  kubectl create -f storageclass-ec.yaml
#################################################################################################################

apiVersion: ceph.rook.io/v1
kind: CephBlockPool
metadata:
  name: replicated-metadata-pool
  namespace: rook-ceph # namespace:cluster
spec:
  failureDomain: osd # host
  replicated:
    size: 3
---
apiVersion: ceph.rook.io/v1
kind: CephBlockPool
metadata:
  name: ec-data-pool
  namespace: rook-ceph # namespace:cluster
spec:
  failureDomain: osd # host
  # Make sure you have enough nodes and OSDs running bluestore to support the replica size or erasure code chunks.
  # For the below settings, you need at least 3 OSDs on different nodes (because the `failureDomain` is `host` by default).
  erasureCoded:
    dataChunks: 3
    codingChunks: 2
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: rook-ceph-block
# Change "rook-ceph" provisioner prefix to match the operator namespace if needed
provisioner: rook-ceph.rbd.csi.ceph.com # driver:namespace:operator
parameters:
  # clusterID is the namespace where the rook cluster is running
  # If you change this namespace, also change the namespace below where the secret namespaces are defined
  clusterID: rook-ceph # namespace:cluster

  # If you want to use erasure coded pool with RBD, you need to create
  # two pools. one erasure coded and one replicated.
  # You need to specify the replicated pool here in the `pool` parameter, it is
  # used for the metadata of the images.
  # The erasure coded pool must be set as the `dataPool` parameter below.
  dataPool: ec-data-pool
  pool: replicated-metadata-pool

  # (optional) mapOptions is a comma-separated list of map options.
  # For krbd options refer
  # https://docs.ceph.com/docs/master/man/8/rbd/#kernel-rbd-krbd-options
  # For nbd options refer
  # https://docs.ceph.com/docs/master/man/8/rbd-nbd/#options
  # mapOptions: lock_on_read,queue_depth=1024

  # (optional) unmapOptions is a comma-separated list of unmap options.
  # For krbd options refer
  # https://docs.ceph.com/docs/master/man/8/rbd/#kernel-rbd-krbd-options
  # For nbd options refer
  # https://docs.ceph.com/docs/master/man/8/rbd-nbd/#options
  # unmapOptions: force

  # RBD image format. Defaults to "2".
  imageFormat: "2"

  # RBD image features. Available for imageFormat: "2". CSI RBD currently supports only `layering` feature.
  imageFeatures: layering

  # The secrets contain Ceph admin credentials. These are generated automatically by the operator
  # in the same namespace as the cluster.
  csi.storage.k8s.io/provisioner-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/provisioner-secret-namespace: rook-ceph # namespace:cluster
  csi.storage.k8s.io/controller-expand-secret-name: rook-csi-rbd-provisioner
  csi.storage.k8s.io/controller-expand-secret-namespace: rook-ceph # namespace:cluster
  csi.storage.k8s.io/node-stage-secret-name: rook-csi-rbd-node
  csi.storage.k8s.io/node-stage-secret-namespace: rook-ceph # namespace:cluster
  # Specify the filesystem type of the volume. If not specified, csi-provisioner
  # will set default as `ext4`.
  csi.storage.k8s.io/fstype: xfs
# uncomment the following to use rbd-nbd as mounter on supported nodes
# **IMPORTANT**: CephCSI v3.4.0 onwards a volume healer functionality is added to reattach
# the PVC to application pod if nodeplugin pod restart.
# Its still in Alpha support. Therefore, this option is not recommended for production use.
#mounter: rbd-nbd
allowVolumeExpansion: true
reclaimPolicy: Delete
EOF
```

### 6) Create a Thorium S3 User

Create a Thorium S3 user and save access/secret key that are generated with the following command. 

```bash
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- radosgw-admin user create --uid=thorium-s3-user --display-name="Thorium S3 User"
```

### 7) Deploy Rook Ceph Toolbox pod

```bash
kubectl https://raw.githubusercontent.com/rook/rook/refs/heads/master/deploy/examples/toolbox.yaml
```

### 8) Verify Rook pods are all running

```bash
kubectl get pods -n rook-ceph
```

For a 5 node k8s cluster with 2 raw storage devices per node, the output might look like this:

```bash
csi-rbdplugin-provisioner-HASH                       5/5     Running     0             1h
csi-rbdplugin-provisioner-HASH                       5/5     Running     0             1h
csi-rbdplugin-HASH                                   3/3     Running     0             1h
csi-rbdplugin-HASH                                   3/3     Running     0             1h
csi-rbdplugin-HASH                                   3/3     Running     0             1h
csi-rbdplugin-HASH                                   3/3     Running     0             1h
csi-rbdplugin-HASH                                   3/3     Running     0             1h
rook-ceph-crashcollector-NODE1-HASH                  1/1     Running     0             1h
rook-ceph-crashcollector-NODE2-HASH                  1/1     Running     0             1h
rook-ceph-crashcollector-NODE3-HASH                  1/1     Running     0             1h
rook-ceph-crashcollector-NODE4-HASH                  1/1     Running     0             1h
rook-ceph-crashcollector-NODE5-HASH                  1/1     Running     0             1h
rook-ceph-exporter-NODE5-HASH                        1/1     Running     0             1h
rook-ceph-exporter-NODE5-HASH                        1/1     Running     0             1h
rook-ceph-exporter-NODE5-HASH                        1/1     Running     0             1h
rook-ceph-exporter-NODE5-HASH                        1/1     Running     0             1h
rook-ceph-exporter-NODE5-HASH                        1/1     Running     0             1h
rook-ceph-mgr-a-HASH                                 3/3     Running     0             1h
rook-ceph-mgr-b-HASH                                 3/3     Running     0             1h
rook-ceph-mon-a-HASH                                 2/2     Running     0             1h
rook-ceph-mon-b-HASH                                 2/2     Running     0             1h
rook-ceph-mon-c-HASH                                 2/2     Running     0             1h
rook-ceph-operator-HASH                              1/1     Running     0             1h
rook-ceph-osd-0-HASH                                 2/2     Running     0             1h
rook-ceph-osd-1-HASH                                 2/2     Running     0             1h
rook-ceph-osd-3-HASH                                 2/2     Running     0             1h
rook-ceph-osd-4-HASH                                 2/2     Running     0             1h
rook-ceph-osd-5-HASH                                 2/2     Running     0             1h
rook-ceph-osd-6-HASH                                 2/2     Running     0             1h
rook-ceph-osd-7-HASH                                 2/2     Running     0             1h
rook-ceph-osd-8-HASH                                 2/2     Running     0             1h
rook-ceph-osd-9-HASH                                 2/2     Running     0             1h
rook-ceph-osd-prepare-NODE5-HASH                     0/1     Completed   0             1h
rook-ceph-osd-prepare-NODE5-HASH                     0/1     Completed   0             1h
rook-ceph-osd-prepare-NODE5-HASH                     0/1     Completed   0             1h
rook-ceph-osd-prepare-NODE5-HASH                     0/1     Completed   0             1h
rook-ceph-osd-prepare-NODE5-HASH                     0/1     Completed   0             1h
rook-ceph-rgw-thorium-s3-store-a-HASH                2/2     Running     0             1h
rook-ceph-tools-HASH                                 1/1     Running     0             1h
```

### 9) Verify Ceph cluster is healthy

If the Rook Ceph cluster is healthy, you should be able to run a status command from the Rook
toolbox. The health section of the cluster status will show `HEALTH_OK`. If you see `HEALTH_WARN`
you will need to look at the reasons at the bottom of the cluster status to troubleshoot the
cause.

```bash
kubectl -n rook-ceph exec -it deploy/rook-ceph-tools -- ceph -s
```

```bash
  cluster:
    id:     20ea7cb0-5cab-4565-bc1c-360b6cd1282b
    health: HEALTH_OK
 
  services:
    mon: 3 daemons, quorum a,b,c (age 1h)
    mgr: b(active, since 1h), standbys: a
    osd: 10 osds: 10 up (since 1h), 10 in (since 1h)
    rgw: 1 daemon active (1 hosts, 1 zones)
 
  data:
...
    ```