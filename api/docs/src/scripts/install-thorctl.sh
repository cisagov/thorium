#!/usr/bin/env bash
set -e

OS=$(uname -s | awk '{print tolower($0)}')
HARDWARE=$(uname -m | sed -r 's/_/-/g')

echo
echo "Detected architecture: \"$OS/$HARDWARE\""

# parse flags
BASE_URL=""
INSECURE=false
TARGET=""
for arg in "$@"; do
  case "$arg" in
    --insecure) INSECURE=true ;;
    --target=*) TARGET="${arg#--target=}" ;;
    *)
      if [ -z "$BASE_URL" ]; then
        BASE_URL="$arg"
      fi
      ;;
  esac
done

if [ -z "$BASE_URL" ]; then
  echo "Usage: install-thorctl.sh <base-url> [--target=<dir>] [--insecure]" >&2
  exit 1
fi

# ask where to store Thorctl at (skip prompt if --target was provided)
if [ -z "$TARGET" ]; then
  echo
  read -p "Enter Thorctl installation directory [~/.local/bin]: " TARGET </dev/tty
  TARGET=${TARGET:-~/.local/bin}
fi

mkdir -p "$TARGET"

URL=$BASE_URL/api/binaries/$OS/$HARDWARE/thorctl

echo
if [ "$INSECURE" = true ]; then
  echo "Downloading Thorctl insecurely from \"$URL\""
  curl -k "$URL" -o "$TARGET/thorctl"
else
  echo "Downloading Thorctl from \"$URL\""
  curl "$URL" -o "$TARGET/thorctl"
fi
chmod +x "$TARGET/thorctl"

if ! [[ ":$PATH:" == *":$TARGET:"* ]]; then
  export PATH=$TARGET:$PATH
  if [ -f "~/.profile" ]; then
    echo "export PATH=$TARGET:\$PATH" >> "~/.profile"
  fi
  echo
  echo "We have added Thorctl to your path but you may need to do it manually."
  echo "If thorctl fails to run add the following to your run commands file (\".bashrc\", \".zshrc\", etc.):"
  echo
  echo "  export PATH=\$PATH:$TARGET"
fi

echo
echo "---------------------------------"
echo
echo "Thorctl installed successfully!"
echo "Run \"thorctl login\" to get started."
echo
echo "For help, run \"thorctl -h\" or see the docs at \"$BASE_URL/api/docs/user/getting_started/thorctl.html\""
echo
