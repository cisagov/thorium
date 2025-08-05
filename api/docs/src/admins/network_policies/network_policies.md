# Network Policies

Thorium network policies provide configurable, fine-grained network isolation for tools running in
Thorium. They are currently exclusive to the Kubernetes Thorium scaler, as they are powered by [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
under the hood. Additionally, a compatible K8's network plugin must also be installed in Thorium's
K8's cluster for policies to actually apply (see the linked K8's docs for more details).

Network policies can only be created, deleted, and updated by Thorium admins. They can be added to
images by tool developers to allow network access to or from tools as needed. Network policies are
also grouped like other resources in Thorium, so only policies in the same group as a tool can
be added to that tool. A tool can have more than one network policy applied at once, and because
network policies are *additive* (they only *add* access rather than removing access), network
policies can never be incompatible with one another.

## Base Network Policies

Network policies can only selectively *allow* network access to or from a tool in Thorium and
don't provide any interface to specifically restrict access. Instead, network access is restricted
in that access is allowed *only* to the entities matching the network policy's (or policies') rules.
Any entity not matching any of the rules is blocked.

That means that if a tool has *no* network policies applied, it will have blanket network access
to and from everything, which defeats the purpose of network policies in the first place. To
mitigate this, Thorium applies one or more "base network policies" to *all* tools running in Thorium,
regardless of their group or which network policies they have already applied. These base network
policies are defined in the configuration file `thorium.yml` (or in the Thorium Custom Resource
Definition when Thorium is deployed with the Thorium Operator). **If no base network policy is given,
a default base policy is applied automatically** that blocks all ingress/egress traffic except traffic
to/from from the Thorium API as well as to the K8's `CoreDNS` and `LocalNodeDNS` services to allow
the Thorium agent to resolve the API's hostname. The default base network policy is provided below
for your reference in [Default Base Network Policy](#default-base-network-policy).

A base network policy must have a unique name among base network policies (but not necessarily unique
to other network policies in Thorium) and a list of ingress/egress rules to apply. Below is an example
base network policy to refer to when creating one in the Thorium config file. The schemas defined
in [Network Policy Schema](#network-policy-schema) may also prove useful.

thorium.yml:

```YAML
...
  - thorium:
    ...
    - base_network_policies:
      - name: "base-policy-1"
        ingress:
          - allowed_ips:
            - cidr: 10.10.10.10
            - cidr: 10.10.0.0/16
              except:
                - 10.10.5.0/24
                - 10.10.6.0/24
            allowed_groups:
              - crabs
              - corn
            allowed_tools:
              - harvester
              - analyzer
            allowed_local: false
            allowed_internet: false
            allowed_all: false
            ports:
              - port: 1000
                end_port: 1006
                protocol: TCP
            allowed_custom:
              - namespace_labels:
                - key: ns-key1
                  value: ns-value1
                - key: ns-key2
                  value: ns-value2
              - pod_labels:
                - key: pod-key1
                  value: pod-value1
                - key: pod-key2
                  value: pod-value2
              - namespace_labels:
                - key: ns-with-pod-key
                  value: ns-with-pod-value
                pod_labels:
                - key: pod-with-ns-key
                  value: pod-with-ns-value
              - <MORE CUSTOM RULES>
          - <MORE RULES>
        egress:
          - <EGRESS RULES>
      - name: "base-policy-2"
        ingress:
          - <RULES>
        egress:
          - <RULES>
```

The base policy should be fairly restrictive to allow other network policies to open up access as
needed. Alternatively, you can bypass Thorium's network policy functionality altogether and allow
full network access for all tools by providing a base network policy with rules to allow all traffic
like below:

```YAML
...
  - thorium:
    ...
    - base_network_policies:
      - name: "allow_all"
        ingress:
          allowed_all: true
        egress:
          allowed_all: true

```

### Default Base Network Policy

If you want to provide other base network policies without overriding the default one, you need to
manually provide the default policy in the Thorium CRD (Custom Resource Definition). Below is the
default base network policy you can copy and paste to the CRD in addition to your custom base
network policies:

```YAML
- name: thorium-default
  ingress:
    - allowed_custom:
      - namespace_labels:
        - key: kubernetes.io/metadata.name
          value: thorium
        pod_labels:
          - key: app
            value: api
  egress:
    - allowed_custom:
      - namespace_labels:
        - key: kubernetes.io/metadata.name
          value: thorium
        pod_labels:
          - key: app
            value: api
    - allowed_ips:
      - cidr: 169.254.0.0/16
      - cidr: fe80::/10
      allowed_custom:
      - namespace_labels:
        - key: kubernetes.io/metadata.name
          value: kube-system
        pod_labels:
          - key: k8s-app
            value: kube-dns
      - namespace_labels:
        - key: kubernetes.io/metadata.name
          value: kube-system
        pod_labels:
          - key: k8s-app
            value: node-local-dns
      ports:
        - port: 53
          protocol: UDP
```

## Network Policy Types

### Forced Network Policies

A forced network policy is forcibly applied in the Thorium scaler to all tools that are in the
policy's groups. Forced policies work similarly to base network policies in that they are not
directly attached to any specific tools and do not appear in an image's info if it wasn't explicitly
added to an image.

### Default Network Policies

A default network policy is a policy that is added to newly-created images in its group(s) when no other policy is provided by the user. Unlike forced network policies, default policies are directly
added to images and will appear in an image's info.

If a network policy is set to no longer be default, it will *not* be automatically removed from the
images it was added to.

## Network Policy Schema

Creating, updating, and managing network policies in Thorium requires an understanding of their
components. Below are a list of fields that make up a network policy, their descriptions, accepted
values, and whether or not the field is required. Use this info to write [rules files](#the-rules-file)
when creating network policies as well as set the base network policy to apply to all tools in a
Thorium instance.

### Network Policy Request

Below are the fundamental components required (or not required) to create a network policy:

| Field | Description | Accepted Values | Required |
| ----- | ----------- | --------------- | -------- |
| name | The name of the network policy | Any UTF-8 string | yes |
| groups | The names of groups the network policy should be in | Any names of groups existing in the Thorium instance | yes |
| ingress | A list of rules applying to ingress traffic into tools; if not provided, all traffic is allowed in; if explicitly set to be empty (no rules), no traffic is allowed in | See [Rules](#rules) | no |
| egress | A list of rules applying to egress traffic from tools; if not provided, all traffic is allowed out; if explicitly set to be empty (no rules), no traffic is allowed out | See [Rules](#rules) | no |
| forced_policy | Sets the policy to apply to all tools spawned in its group(s); forced policies are not actually saved to individual images and are applied in the scaler when images are spawned | true/false (default: false) | no |
| default_policy | Sets the policy to be added by default to an image on creation if no other policies are given; default policies are actually saved to all new images in their groups | true/false (default: false) | no |

### Rules

A network policy rule dictates which specific entities a Thorium image can connect to or be connected
from. Rules are *additive*, meaning they combine together and can never deny what another rule allows.
If one rule allows ingress access from the "corn" group, no other rule can deny access from that group.
This also means policy rules are never incompatible with each other.

| Field | Description | Accepted Values | Required |
| ----- | ----------- | --------------- | -------- |
| allowed_ips | A list of IP's to allow | See [IP Blocks](#ip-blocks) | no |
| allowed_groups | A list of groups to allow | A name of any group existing in the Thorium instance | no |
| tools | A list of tools to allow | Any valid tool name | no |
| allowed_local | Allows all IP addresses in the local IP address space access | true/false (default: false) | no |
| allowed_internet | Allows all IP addresses in the public IP address space access | true/false (default: false) | no |
| allowed_all | Allows from all entities | true/false (default: false) | no |
| ports | A list of ports this rule applies to; if not provided, the rule will apply on all ports | See [Ports](#ports) | no |
| allowed_custom | A list of custom rules allowing access to peers on K8's matched by namespace and/or pod label(s) | See [K8's Custom Rules](#k8s-custom-rules) | no |

### IP Blocks

An IP block defines one or more IP addresses to allow access to or from. They can be defined as a
simple IP address or as an IP CIDR covering an address space. In the later case, an optional list of
CIDR's can be provided to exclude certain addresses access within an address space.

| Field | Description | Valid Values | Required |
| ----- | ----------- | ------------ | -------- |
| cidr | A IPv4 or IPv6 CIDR or a single IPv4 or IPv6 address to allow | A valid IPv4/IPv6 CIDR or address | yes |
| except | A list of CIDR's to exclude from the allowed CIDR described above | Zero or more CIDR's within the allowed CIDR described above; an error occur if any of the CIDR's are not in the allowed CIDR's address space, are of a different standard (v4 vs v6), or if the cidr above is a single IP address and except CIDR's were provided | no |

### Ports

Ports limit the scope of a given network policy rule to a single port or a range of ports and
optionally a specific protocol.

For example, if a user wanted to allow access to port 80 over TCP from any entity, they could provide
an ingress rule with `allowed_all=true` and a port rule with `port=80` and `protocol=TCP`. If a user
wanted to allow tools to access ports 1000-1006 over any protocol but only to tools in the "corn"
group, they could provide an egress rule with `allowed_groups=["corn"]` and a port rule with
`port=1000`, `end_port=1006`, and no value set for `protocol`.

| Field | Description | Valid Values | Required |
| ----- | ----------- | ------------ | -------- |
| port | The port to allow, or the first port in a range of ports to allow when used in conjunction with `end_port` | Any valid port number (1-65535) | yes |
| end_port | The last port in the range of ports starting with `port` | Any valid port number (1-65535) | no |
| protocol | The protocol to allow on the specified port(s); if not provided, all protocols are allowed | TCP/UDP/SCTP | no |

### K8's Custom Rules

K8's custom rules provide fine-grained control to allow tool access to or from entities in the K8's cluster
that aren't in Thorium. You can provide namespace labels to match for entire namespaces or
pod labels to match for specific pods. If both namespace *and* pod labels are specified, only
pods with all of the given pod labels that are in a namespace with all of the
given namespace labels will match.

| Field | Description | Accepted Values | Required |
| ----- | ----------- | --------------- | -------- |
| namespace_labels | A list of labels matching namespaces to allow | See [K8's Custom Labels](#k8s-custom-labels) | no |
| pod_labels | A list of labels matching pods to allow | See [K8's Custom Labels](#k8s-custom-labels) | no |

### K8's Custom Labels

K8's custom labels will match K8's resources with the given key/value pairs.

| Field | Description | Accepted Values | Required |
| ----- | ----------- | --------------- | -------- |
| key | The label key to match on | Any valid K8's label name (see the [K8's docs](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set)) | yes |
| value | The label value to match on | Any valid K8's label name (see the [K8's docs](https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#syntax-and-character-set)) | yes |
