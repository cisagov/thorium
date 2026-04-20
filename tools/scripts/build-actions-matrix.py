#!/usr/bin/env python3

import argparse
import json
import logging
import sys


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


def config_image_tags(config):
    """Extract possible container image tags from an embedded image config.

    The toolbox spec uses config.image for the primary container image URL.
    This also tolerates config.images if present in a future/custom shape.
    """
    if not isinstance(config, dict):
        return []

    tags = []

    image = config.get("image")
    if isinstance(image, str):
        tags.append(image)

    images = config.get("images")
    if isinstance(images, str):
        tags.append(images)
    elif isinstance(images, list):
        tags.extend(item for item in images if isinstance(item, str))

    return tags


def load_toolbox_json(path):
    """Load generated toolbox.json only."""
    if not path.endswith(".json"):
        logging.error(f"Unsupported manifest type for {path}; expected toolbox.json")
        sys.exit(2)

    try:
        with open(path, "r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)
    except FileNotFoundError:
        logging.error(f"Failed to find manifest: {path}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON manifest file {path} with error {e}")
        sys.exit(2)


def build_matrix(toolbox_manifest):
    """Build a GitHub Actions matrix array from toolbox.json."""
    images = toolbox_manifest.get("images")
    if not isinstance(images, dict):
        logging.error("No images object in toolbox manifest")
        sys.exit(2)

    matrix = []
    fatal_errors = []

    for image_name, versions in images.items():
        if not isinstance(versions, dict):
            fatal_errors.append(f'Image "{image_name}" has invalid versions object')
            continue

        for version, image_entry in versions.items():
            if not isinstance(image_entry, dict):
                fatal_errors.append(
                    f'Image "{image_name}" version "{version}" has invalid entry object'
                )
                continue

            # image_from entries are not built. In valid toolbox.json, these should
            # also have build_image = false, but skip explicitly for clarity.
            if "image_from" in image_entry:
                continue

            # build_image false is intentionally skipped quietly.
            if not image_entry.get("build_image", True):
                continue

            build_path = image_entry.get("build_path")
            if not isinstance(build_path, str) or not build_path.strip():
                fatal_errors.append(
                    f'Image "{image_name}" version "{version}" has build_image=true '
                    f"but missing/empty build_path"
                )
                continue

            image_tags = image_entry.get("image_tags", [])
            if image_tags is None:
                image_tags = []

            if not isinstance(image_tags, list):
                fatal_errors.append(
                    f'Image "{image_name}" version "{version}" has non-list image_tags'
                )
                continue

            config = image_entry.get("config", {})
            tags = unique_preserve_order(
                list(image_tags) + config_image_tags(config)
            )

            if not tags:
                fatal_errors.append(
                    f'Image "{image_name}" version "{version}" has build_image=true '
                    f"and build_path={build_path!r}, but no image_tags or config.image"
                )
                continue

            base_image = image_entry.get("base_image", {})
            if base_image is None:
                base_image = {}

            if not isinstance(base_image, dict):
                fatal_errors.append(
                    f'Image "{image_name}" version "{version}" has invalid base_image; '
                    f"expected object"
                )
                continue

            matrix.append(
                {
                    "name": image_name,
                    "version": version,
                    "build_path": build_path,
                    # Backward-compatible key expected by older workflows.
                    "image_name": tags[0],
                    "image_tags": tags,
                    # Pass through only the per-entry toolbox image-level base_image object.
                    # The build script interprets allow_override/image/image_arg/token/user.
                    "base_image": base_image,
                }
            )

    if fatal_errors:
        for error in fatal_errors:
            logging.error(error)
        sys.exit(2)

    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Generate a GitHub Actions image-build matrix from toolbox.json"
    )
    parser.add_argument(
        "manifest",
        help="Path to generated toolbox.json",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    toolbox_manifest = load_toolbox_json(args.manifest)
    matrix = build_matrix(toolbox_manifest)

    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
