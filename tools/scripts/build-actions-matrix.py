import logging
import json
import tomllib
import sys


if (len(sys.argv) != 2):
    print(f"Usage: python3 {sys.argv[0]} toolbox.json")
    exit(1)

manifest_path = sys.argv[1]
toolbox_manifest = {}
try:
    with open(manifest_path, 'r') as manifest_file:
        if (manifest_path.endswith(".json")):
            toolbox_manifest = json.load(manifest_file)
        elif (manifest_path.endswith(".toml")):
            toolbox_manifest = tomllib.load(manifest_file)
        logging.info(f"Loaded manifest from file {manifest_path}")
except FileNotFoundError:
    logging.error(f"Failed to find manifest: {manifest_path}")
    exit(1)
except json.JSONDecodeError as e:
    logging.error(f"Failed to decode JSON manifest file ${manifest_path} with error {e}")
    exit(1)
except tomllib.TOMLDecodeError as e:
    logging.error(f"Failed to decode TOML manifest file ${manifest_path} with error {e}")
    exit(1)

images = toolbox_manifest.get("images", None)
if images is None:
    logging.error('No images in manifest, existing')
    exit(1)

matrix = []
# loop through each image name
for image in images.keys():
    # get each image version
    for version in images.get(image, {}):
        if not images[image][version].get("build_image", True):
            continue
        # grab build path for docker context for image/version
        build_path = images[image][version].get("build_path")
        image_tags = images[image][version].get("image_tags", [])
        # grab registry image name from Thorium image config
        config = images[image][version].get("config", {})
        image_name = config.get("image", "")
        # both context build_path and registry image_name must be specified
        if (build_path != "" and image_name != ""):
            matrix.append({"build_path": build_path, "image_name": image_name, "image_tags": image_tags})
        else:
            logging.error(f"Image \"{config.get("name", "")}\" with empty build_path ({build_path}) or image name ({image_name})")

# no dump matrix to std
print(json.dumps(matrix))
