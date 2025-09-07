
# Overview

This folder contains a set of Ansible playbooks for deploying Thorium on top of locally hosted baremetal servers or VMs. This assumes you have no access to external storage interfaces or other cloud native storage/dbs. That said you can repurpose the `deploy.yml` playbook and inventory group variables (`inventory/group_vars`) for hosted environments by disabling roles that are made redundant by other services within your environment. You may have to conduct some steps manually, such as creating Elastic indexes when running this in hosted environments.

### Prerequisites

The host you run these Ansible roles from will need network connectivity to your Kubernetes cluster kube-api service along with a valid kube config in your home directory. You will also need an installation of python3 and pip.

Install prerequisites 

 - python3
 - pip

```bash
python3 -m pip install -r ansible kubernetes boto3
ansible-galaxy collection install kubernetes.core amazon.aws --upgrade
```

Alternatively you can package up all python dependencies using a virtual environment, this is most useful for using this playbook in environments with restricted internet access.

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
python3 -m pip install -r ansible kubernetes boto3
ansible-galaxy collection install kubernetes.core amazon.aws --upgrade
```

Ansible galaxy modules are installed in `~/.ansible` and would need to be saved when conducting offline deployments.

### Kernel Parameters

Some systems may require setting certain kernel parameters for systems like Elastic to startup properly.

```
fs.aio-max-nr=2097152
fs.file-max=8097152
vm.max_map_count=262144
```

### File Staging (offline usage only)

```bash
ansible-playbook -i inventory/local.ini offline-stage.yml -v
```

For offline deploments, stage any nessesary files into the `./files` directory. These files will get moved up along with the full playbook. 

### Usage

Update the `group_vars` located in `inventories/group_vars` to match the requirements of your environment. This playbook generates all secret key/passwords unless specified in the inventory variables. An ansible vault is generated containing those secret values to ensure that susequent runs of the playbook do not regenerate/override those values. From one of the target k8s master nodes or a server with network connectivity to the kubeapi and a valid kube config, deploy thorium and database dependencies on top of k8s:

```bash
ansible-playbook -i inventories/local.ini deploy.yml --ask-vault-pass -v
```

or

```bash
ansible-playbook -i inventories/local.ini deploy.yml --vault-pass-file artifacts/vault_pass --extra-vars "vault_password_file=artifacts/vault_pass" -v
```

### Cleanup

This section documents commands for cleaning up k8s resources that are deployed by these ansible roles. Cleanup can be helpful for testing the playbooks without redploying your k8s environment. Delete resources in the following order while confirming all the resources from one section have been deleted before moving to the next section.

Cleanup Thorium and DBs/elastic

```
kubectl delete ThoriumCluster dev -n thorium
helm uninstall -n traefik traefik
helm uninstall -n quickwit quickwit
kubectl delete statefulset -n redis redis
kubectl delete kibana -n elastic-system elastic
kubectl delete elasticsearch -n elastic-system elastic
kubectl delete Kubegres -n quickwit postgres
kubectl delete scyllacluster -n scylla scylla
kubectl delete statefulset.apps/jaeger -n jaeger
```

Remove operators and cert-manager

```bash
helm uninstall -n scylla-operator scylla
kubectl delete deployment -n thorium operator
kubectl delete deployment -n kubegres-system kubegres-controller-manager
kubectl delete deployment.apps/cert-manager -n cert-manager
kubectl delete deployment.apps/cert-manager-cainjector -n cert-manager
kubectl delete deployment.apps/cert-manager-webhook -n cert-manager
kubectl delete statefulset.apps/elastic-operator -n elastic-system
```

Delete PVCs that remain after stateful resources have been deleted. Note: multiple PVCs may exist for multi-node k8s clusters that have deployed scaled up DBs.

```bash
# kubectl delete pvc -n redis redis-persistent-storage-claim
# kubectl delete pvc -n scylla data-scylla-us-east-1-us-east-1a-0
# kubectl delete pvc -n elastic-system elasticsearch-data-elastic-es-default-0
# kubectl delete pvc -n quickwit postgres-db-postgres-1-0
```
