kubectl describe pods -A | grep Image | grep -v ID | grep -v kubelet | awk '{print $2}' | sort | uniq
