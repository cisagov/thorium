ARG IMAGE="python:3.11-slim"
FROM $IMAGE

RUN pip install uefi_firmware

WORKDIR /app/
COPY uefi-firmware-parser.py ./
COPY wrapper.bash ./

# set command
ENTRYPOINT ["/app/wrapper.bash"]