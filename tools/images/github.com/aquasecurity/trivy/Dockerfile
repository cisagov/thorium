ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

RUN apt update -y && \
  apt-get install -y wget gnupg && \
  wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | gpg --dearmor | tee /usr/share/keyrings/trivy.gpg > /dev/null && \
  echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb generic main" | tee -a /etc/apt/sources.list.d/trivy.list && \
  apt-get update -y && \
  apt-get install -y trivy

# prefetch and save our DB
RUN trivy fs --scanners vuln,secret,misconfig,license /tmp && \
    mv /root/.cache/* /tmp/cache/ && \
    chmod -R 777 /tmp/cache

# Add our wrapper.sh script
WORKDIR /app
COPY wrapper.sh .

# setup our entrypoint
ENTRYPOINT ["/app/wrapper.sh"]