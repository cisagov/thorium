FROM ghcr.io/fkie-cad/cwe_checker:latest

WORKDIR /home/cwe

### REPO: https://github.com/fkie-cad/cwe_checker 
ADD wrapper.sh /home/cwe/.

ENTRYPOINT ["bash", "/home/cwe/wrapper.sh"]