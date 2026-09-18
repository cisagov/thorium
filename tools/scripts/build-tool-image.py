#!/usr/bin/env python3

import argparse
import json
import logging
import os
import subprocess
import sys


def run_command(command, *, dry_run=False):
    """Run a command, printing it first."""
    printable = " ".join(command)
    logging.info("+ %s", printable)

    if dry_run:
        return

    subprocess.run(command, check=True)


def parse_json_arg(value, default):
    """Parse a JSON CLI argument."""
    if value is None or value == "":
        return default

    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        logging.error("Failed to parse JSON argument %r: %s", value, e)
        sys.exit(2)


def unique_preserve_order(values):
    """Return unique non-empty string values while preserving order."""
    seen = set()
    output = []

    for value in values:
        if not isinstance(value, str):
            continue

        value = value.strip()
        if not value:
            continue

        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def strip_var_syntax(value):
    """Convert '$TOKEN' or '${TOKEN}' to 'TOKEN'.

    If value is already 'TOKEN', it is returned unchanged.
    """
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    if value.startswith("${") and value.endswith("}"):
        return value[2:-1]

    if value.startswith("$"):
        return value[1:]

    return value


def env_value_from_var_name(value, *, field_name):
    """Resolve a toolbox pass-through variable name to its environment value."""
    var_name = strip_var_syntax(value)
    if not var_name:
        return None

    resolved = os.environ.get(var_name)
    if resolved is None:
        logging.error(
            "%s references environment variable %r, but it is not set",
            field_name,
            var_name,
        )
        sys.exit(2)

    return resolved


def get_registry_from_image(image_name):
    """Return explicit registry component from a container image reference.

    Examples:
      ghcr.io/example-org/image:tag                 -> ghcr.io
      registry.example.com/project/image:tag        -> registry.example.com
      localhost:5000/image:tag                      -> localhost:5000
      ubuntu:24.04                                  -> None
    """
    if not image_name or not isinstance(image_name, str):
        return None

    first_component = image_name.split("/")[0]

    if (
        "." in first_component
        or ":" in first_component
        or first_component == "localhost"
    ):
        return first_component

    return None


def parse_registry_login(value):
    """Parse --registry-login REGISTRY=USER_ENV=TOKEN_ENV."""
    try:
        registry, user_env, token_env = value.split("=", 2)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--registry-login must be REGISTRY=USER_ENV=TOKEN_ENV"
        )

    if not registry:
        raise argparse.ArgumentTypeError("--registry-login REGISTRY cannot be empty")
    if not user_env:
        raise argparse.ArgumentTypeError("--registry-login USER_ENV cannot be empty")
    if not token_env:
        raise argparse.ArgumentTypeError("--registry-login TOKEN_ENV cannot be empty")

    return {
        "registry": registry,
        "user_env": user_env,
        "token_env": token_env,
    }


def registry_login_map(registry_logins):
    """Convert parsed registry login entries to registry -> creds mapping."""
    output = {}

    for entry in registry_logins or []:
        output[entry["registry"]] = {
            "user_env": entry["user_env"],
            "token_env": entry["token_env"],
        }

    return output


def docker_login(runtime, registry, username, token, *, dry_run=False):
    """Log in to a container registry using password-stdin."""
    logging.info("Logging in to registry %s as %s", registry, username)

    if dry_run:
        logging.info(
            "+ echo *** | %s login -u %s --password-stdin %s",
            runtime,
            username,
            registry,
        )
        return

    login = subprocess.run(
        [runtime, "login", "-u", username, "--password-stdin", registry],
        input=token,
        text=True,
        check=False,
    )

    if login.returncode != 0:
        logging.error("Failed to login to registry %s", registry)
        sys.exit(login.returncode)


def login_to_base_registry(runtime, base_image_config, *, dry_run=False):
    """Log in to the base image registry if base_image token/user are present.

    This is independent of allow_override. token/user are variable names, not
    raw secret values.
    """
    if not isinstance(base_image_config, dict):
        logging.error("base_image must be a JSON object")
        sys.exit(2)

    token_var = base_image_config.get("token")
    user_var = base_image_config.get("user")

    if not token_var and not user_var:
        return

    if not token_var or not user_var:
        logging.error(
            "base_image.token and base_image.user must both be set to login "
            "to the base image registry"
        )
        sys.exit(2)

    base_image = base_image_config.get("image")
    base_registry = get_registry_from_image(base_image)

    if not base_registry:
        logging.warning(
            "base_image.token/user are set, but base_image.image has no explicit registry; "
            "skipping base registry login"
        )
        return

    username = env_value_from_var_name(user_var, field_name="base_image.user")
    token = env_value_from_var_name(token_var, field_name="base_image.token")

    docker_login(
        runtime,
        base_registry,
        username,
        token,
        dry_run=dry_run,
    )


def login_to_destination_registries(runtime, image_tags, registry_logins, *, dry_run=False):
    """Log in to destination registries using explicitly supplied mappings.

    Tags whose registries are not in registry_logins are assumed to already be
    authenticated or public.
    """
    logins = registry_login_map(registry_logins)

    registries = unique_preserve_order(
        get_registry_from_image(tag) for tag in image_tags
    )

    for registry in registries:
        if not registry:
            continue

        login_config = logins.get(registry)
        if login_config is None:
            logging.info(
                "No --registry-login mapping provided for destination registry %s; "
                "assuming already authenticated or public",
                registry,
            )
            continue

        user_env = login_config["user_env"]
        token_env = login_config["token_env"]

        username = os.environ.get(user_env)
        token = os.environ.get(token_env)

        if username is None:
            logging.error(
                "Destination registry %s user env var %r is not set",
                registry,
                user_env,
            )
            sys.exit(2)

        if token is None:
            logging.error(
                "Destination registry %s token env var %r is not set",
                registry,
                token_env,
            )
            sys.exit(2)

        docker_login(
            runtime,
            registry,
            username,
            token,
            dry_run=dry_run,
        )


def base_build_arg(base_image_config):
    """Return KEY=VALUE base image build arg, or None.

    Rules:
      - no base_image.image -> no substitution
      - allow_override defaults true
      - allow_override false -> no substitution
      - image_arg defaults IMAGE
    """
    if not isinstance(base_image_config, dict):
        logging.error("base_image must be a JSON object")
        sys.exit(2)

    allow_override = base_image_config.get("allow_override", True)
    if allow_override is False:
        return None

    image = base_image_config.get("image")
    if not image:
        return None

    image_arg = base_image_config.get("image_arg") or "IMAGE"

    return f"{image_arg}={image}"


def build_tag(
    runtime,
    build_path,
    image_tag,
    base_arg,
    labels,
    *,
    pull=False,
    no_cache=False,
    dry_run=False,
):
    """Build the primary image tag."""
    dockerfile = os.path.join(build_path, "Dockerfile")

    command = [
        runtime,
        "build",
        "--file",
        dockerfile,
        "--tag",
        image_tag,
    ]

    for label in labels:
        command.extend(["--label", label])

    if base_arg is not None:
        command.extend(["--build-arg", base_arg])

    if pull:
        command.append("--pull")

    if no_cache:
        command.append("--no-cache")

    command.append(build_path)

    run_command(command, dry_run=dry_run)


def tag_image(runtime, source_tag, destination_tag, *, dry_run=False):
    """Retag one image."""
    if source_tag == destination_tag:
        return

    run_command(
        [runtime, "tag", source_tag, destination_tag],
        dry_run=dry_run,
    )


def push_tag(runtime, image_tag, *, dry_run=False):
    """Push one image tag."""
    run_command(
        [runtime, "push", image_tag],
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build, retag, and push one toolbox image from a GitHub Actions matrix entry"
    )

    parser.add_argument(
        "--build-path",
        required=True,
        help="Docker build context path",
    )

    parser.add_argument(
        "--image-tags",
        required=True,
        help=(
            "JSON list of image tags. First tag is used for the build; "
            "others are aliases applied with tag."
        ),
    )

    parser.add_argument(
        "--base-image",
        default="{}",
        help="JSON base_image object from toolbox.json image entry",
    )

    parser.add_argument(
        "--container-runtime",
        default="docker",
        help="Container runtime to use, usually docker or podman",
    )

    parser.add_argument(
        "--registry-login",
        action="append",
        type=parse_registry_login,
        default=[],
        help=(
            "Destination registry login mapping in REGISTRY=USER_ENV=TOKEN_ENV form. "
            "Example: ghcr.io=GITHUB_ACTOR=GITHUB_TOKEN"
        ),
    )

    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label to add to the image build. May be repeated.",
    )

    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pass --pull to the image build",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Pass --no-cache to the image build",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    image_tags = parse_json_arg(args.image_tags, default=[])
    if not isinstance(image_tags, list):
        logging.error("--image-tags must be a JSON list")
        sys.exit(2)

    image_tags = unique_preserve_order(image_tags)
    if not image_tags:
        logging.error("--image-tags must contain at least one tag")
        sys.exit(2)

    base_image_config = parse_json_arg(args.base_image, default={})
    if not isinstance(base_image_config, dict):
        logging.error("--base-image must be a JSON object")
        sys.exit(2)

    build_path = args.build_path
    if not build_path:
        logging.error("--build-path is required")
        sys.exit(2)

    dockerfile = os.path.join(build_path, "Dockerfile")
    if not os.path.exists(dockerfile):
        logging.error("Dockerfile not found: %s", dockerfile)
        sys.exit(2)

    primary_tag = image_tags[0]

    # Login to the base registry first. This is independent of allow_override.
    login_to_base_registry(
        args.container_runtime,
        base_image_config,
        dry_run=args.dry_run,
    )

    # Login to destination registries when mappings are provided.
    login_to_destination_registries(
        args.container_runtime,
        image_tags,
        args.registry_login,
        dry_run=args.dry_run,
    )

    base_arg = base_build_arg(base_image_config)

    build_tag(
        args.container_runtime,
        build_path,
        primary_tag,
        base_arg,
        args.label,
        pull=args.pull,
        no_cache=args.no_cache,
        dry_run=args.dry_run,
    )

    for tag in image_tags[1:]:
        tag_image(
            args.container_runtime,
            primary_tag,
            tag,
            dry_run=args.dry_run,
        )

    for tag in image_tags:
        push_tag(
            args.container_runtime,
            tag,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
