ARG IMAGE="python:3.11-slim"
FROM $IMAGE

WORKDIR "/app"

# the capa release to install; capa-rules and the capa signatures are pinned to the matching release
# renovate: datasource=github-releases depName=mandiant/capa extractVersion=^v(?<version>.+)$
ARG CAPA_VERSION="9.4.0"
# renovate: datasource=git-refs depName=https://github.com/mandiant/capa currentValue=v${CAPA_VERSION}
ARG CAPA_COMMIT="7a79f799a70f052e8382cdbeaec8115daba14a5b"
# renovate: datasource=git-refs depName=https://github.com/mandiant/capa-rules currentValue=v${CAPA_VERSION}
ARG CAPA_RULES_COMMIT="2af9fbfc1c9b4634dbeb76b5d34fca9389fa7f80"

RUN pip install --no-cache-dir "flare-capa==${CAPA_VERSION}"

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# fetch the pinned capa-rules and capa commits and verify them
RUN git init -q /app/capa-rules && \
    git -C /app/capa-rules fetch -q --depth 1 https://github.com/mandiant/capa-rules.git "$CAPA_RULES_COMMIT" && \
    git -C /app/capa-rules checkout -q --detach FETCH_HEAD && \
    test "$(git -C /app/capa-rules rev-parse HEAD)" = "$CAPA_RULES_COMMIT" && \
    git init -q /app/capa && \
    git -C /app/capa fetch -q --depth 1 https://github.com/mandiant/capa.git "$CAPA_COMMIT" && \
    git -C /app/capa checkout -q --detach FETCH_HEAD && \
    test "$(git -C /app/capa rev-parse HEAD)" = "$CAPA_COMMIT"

COPY . /app/.

ENTRYPOINT ["bash", "/app/wrapper.bash"]
