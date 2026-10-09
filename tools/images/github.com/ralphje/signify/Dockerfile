ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

# pin signify (pe_certs.py uses the 0.9 API) and mscerts (2026.8.28 has a trust
# store entry that asn1crypto 1.5.1 cannot parse, so verify() raises KeyError)
# renovate: datasource=pypi depName=signify
ARG SIGNIFY_VERSION=0.9.2
# renovate: datasource=pypi depName=mscerts allowedVersions=<2026.8.28
ARG MSCERTS_VERSION=2026.7.1
RUN pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir "signify==${SIGNIFY_VERSION}" "mscerts==${MSCERTS_VERSION}" pyasn1

COPY . /app/.

ENTRYPOINT ["python3", "/app/run_signify.py"]
