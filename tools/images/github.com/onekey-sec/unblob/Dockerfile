FROM ghcr.io/onekey-sec/unblob:latest

# container was built as unblob user, this causes issues
# as other issues for python paths and directory perms
WORKDIR /home/unblob
COPY wrapper.sh /home/unblob/.

# set command
ENTRYPOINT ["/bin/bash", "wrapper.sh"]