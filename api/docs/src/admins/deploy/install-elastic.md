## Deploy ECK

> The Elasticsearch deployment process will be summarized in this guide. However, admins may want
> to consult the official ECK documentation for a more complete explanation of different configuration
> options and additional troubleshooting steps. See the [Elasticsearch deployment guide](https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-deploy-eck.html).

Thorium requires `Elasticsearch` to enable full-text search of analysis tool results and other
submission data. `Kibana` may optionally be deployed as a web interface for managing the ECK
configuration such as user roles, permissions and storage indexes.

### 1) Deploy Elastic Operator and CRDs

> Please consult the [supported versions](https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-supported.html)
> of the `Elastic Cloud on Kubernetes` documentation to ensure the operator supports the Kubernetes
> version of your environment as well as the Elasticsearch and Kibana version you will be deploying.

Create an Elasticsearch operator and the related CRDs. It may be necessary to update the following
command with the latest operator and CRD version. Note that the shared operator/crd version will
differ from the `Elasticsearch` and `Kibana` version.

```bash
kubectl apply -f https://download.elastic.co/downloads/eck/2.16.1/crds.yaml
kubectl create -f https://download.elastic.co/downloads/eck/2.16.1/operator.yaml
```

### 2) Update System Configuration

The system configuration of K8s cluster nodes may need to be updated to meet the resource requirements
of ECK. In particular the maximum allowed virtual memory maps for an individual process must be increased
for elastic pods to successfully start. This configuration value may be added to a linux system's 
`/etc/sysctl.conf` or `/etc/sysctl.d/99-sysctl.conf` file. Be aware that some linux versions ignore the
 `/etc/sysctl.conf` file on boot.

```bash
echo "vm.max_map_count=262144" >> /etc/sysctl.d/99-sysctl.conf
```

See [system configuration options for elastic nodes](https://www.elastic.co/guide/en/elasticsearch/reference/current/system-config.html). You can also set an `initContainer` to run before elastic starts that will set the `max_map_count`. That option is what the next step will show.

### 3) Deploy Kibana and ElasticSearch

You may want to update these fields in the following resource files before applying them with `kubectl`:

- `version` - version of ES and Kibana you want to deploy
- `count` - number of nodes in your ES cluster or kibana replicas
- `storageClassName` - name of the storage provisioner for requesting K8s PVCs
- `resources.requests.storage` - size of storage volumes for each ES pod
- `resources.[requests,limits].memory` - memory for each ES and Kibana pod
- `resources.[requests,limits].cpu` - cpu for each ES and Kibana pod

#### Deploy Elastic
```yaml,editable
cat <<EOF | kubectl apply -f -
apiVersion: elasticsearch.k8s.elastic.co/v1
kind: Elasticsearch
metadata:
  name: elastic
  namespace: elastic-system
spec:
  version: 8.17.2
  volumeClaimDeletePolicy: DeleteOnScaledownOnly
  nodeSets:
  - name: default
    count: 3
    podTemplate:
      spec:
        initContainers:
        - name: sysctl
          securityContext:
            privileged: true
            runAsUser: 0
          command: ['sh', '-c', 'sysctl -w vm.max_map_count=262144']
        containers:
        - name: elasticsearch
          env:
          - name: ES_JAVA_OPTS
            value: -Xms28g -Xmx28g
          resources:
            requests:
              memory: 32Gi
              cpu: 4
            limits:
              memory: 32Gi
              cpu: 4
    volumeClaimTemplates:
      - metadata:
          name: elasticsearch-data
        spec:
          storageClassName: csi-rbd-sc
          accessModes:
          - ReadWriteOnce
          resources:
            requests:
              storage: 12Ti
    config:
      node.store.allow_mmap: true
      http.max_content_length: 1024mb
EOF
```

#### Deploy `Kibana`

```yaml,editable
cat <<EOF | kubectl apply -f -
apiVersion: kibana.k8s.elastic.co/v1
kind: Kibana
metadata:
  name: elastic
  namespace: elastic-system
spec:
  version: 8.17.2
  count: 1
  elasticsearchRef:
    name: elastic
EOF
```

### 4) Verify Elastic and Kibana are up

Ensure the Elastic and Kibana pods are `Running`.

```bash
kubectl get pods -n elastic-system
# NAME                          READY   STATUS    RESTARTS       AGE
# elastic-es-default-0          1/1     Running   0              1h
# elastic-es-default-1          1/1     Running   0              1h
# elastic-es-default-2          1/1     Running   0              1h
# elastic-kb-55f49bdfb4-p6kg9   1/1     Running   0              1h
# elastic-operator-0            1/1     Running   0              1h
```

### 5) Create Thorium role and index

Create the Elastic thorium user, role, and results index using the following command. Be sure to update the
`INSECURE_ES_PASSWORD` to an appropriately secure value. The text block can be edited before copy-pasting
the command into a terminal.

> You will use the username, password, and index name configured here when you create the
> ThoriumCluster resource.

```bash,editable
export ESPASS=$(kubectl get secret -n elastic-system elastic-es-elastic-user -o=jsonpath='{.data.elastic}' | base64 --decode; echo)
kubectl -n elastic-system exec -i --tty=false pod/elastic-es-default-0 -- /bin/bash << EOF
# Create thorium role
curl -k -X PUT -u elastic:$ESPASS "https://localhost:9200/thorium?pretty"
# Create results index and give thorium role privileges
curl -k -X POST -u elastic:$ESPASS "https://localhost:9200/_security/role/thorium?pretty" -H 'Content-Type: application/json' -d'
{
  "indices": [
    {
      "names": ["results"],
      "privileges": ["all"]
    }
  ]
}
'
# Create thorium user with thorium role
curl -k -X POST -u elastic:$ESPASS "https://localhost:9200/_security/user/thorium?pretty" -H 'Content-Type: application/json' -d'
{
  "password" : "INSECURE_ES_PASSWORD",
  "roles" : ["thorium"],
  "full_name" : "Thorium",
  "email" : "thorium@sandia.gov"
}
'
EOF
```