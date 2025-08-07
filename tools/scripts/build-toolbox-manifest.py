import sys
import tomllib #tomllib from std only supports load/loads, toml supports dump but has awful formatting
import json
import os
import logging

if (len(sys.argv) != 2):
    logging.error(f"Usage: python3 {sys.argv[0]} config.toml")
    exit(1)

# create empty market place manifest
toolbox_manifest = {
    "pipelines": {},
    "images": {},
}
try:
    # load toolbox manifest file
    toolbox_config_path = sys.argv[1]
    with open(toolbox_config_path, 'rb') as config_file:
        config = tomllib.load(config_file)
    toolbox_manifest["name"] = config.get("name", "")
    toolbox_manifest["registry"] = config.get("registry", "")
except FileNotFoundError:
    logging.error(f"Failed to find toolbox manifest {toolbox_config_path}")
    exit(1)
except tomllib.TOMLDecodeError as e:
    logging.error(f"Failed to decode TOML toolbox config ${toolbox_config_path} with error: {e}")
    exit(1)
except Exception as e:
    logging.error(f"Failed to load toolbox config with unexpected error: {e}")
    exit(1)

def image_fields(manifest):
    image = {}
    # Registry image name
    name = manifest.get("name", "")
    image_name = manifest.get("image_name", "")
    image_version = manifest.get("version", "")
    build_image = manifest.get("build", True)
    if (image_version == ""):
        logging.error(f" No image version found for {name}")
    if (image_name == ""):
        logging.error(f" No image name found for {name}")
    else:
        image_name = f"{toolbox_manifest["registry"]}/{image_name}:{image_version}"

    # Container build context path
    image_build_path = f"{root}"
    if (manifest["build_path"] not in ['./', '.']):
        image_build_path += f"/{manifest["build_path"]}"
    image["build_path"] = image_build_path
    image["build_image"] = build_image
    # Thorium image configuration
    if ('config_from' in manifest and 'config' not in manifest):
        config_path = f"{root}/{manifest['config_from']}"
        with open(config_path, 'rb') as config_file:
            config = json.load(config_file)
            config["image"] = image_name
            image["config"] = config
    return image

def pipeline_fields(manifest):
    pipeline = {}
    pipeline["description"] = manifest.get("description", "")
    pipeline["images"] = manifest.get("images", {})
    if ('config_from' in manifest and 'config' not in manifest):
        config_path = f"{root}/{manifest['config_from']}"
        with open(config_path, 'rb') as config_file:
            config = json.load(config_file)
            pipeline["config"] = config
    return pipeline

images = dict()
pipelines = dict()
# loop through any directories looking for manifests
for root, dirs, files in os.walk('.'):
    for file in files:
        if (file == 'manifest.toml' and root != '.'):
            manifest_path = os.path.join(root, file)
            with open(manifest_path, 'rb') as manifest_file:
                # read in the TOML formatted manifest file
                manifest = tomllib.load(manifest_file)
                # import this pipeline to the toolbox manifest
                if manifest["type"] == "pipeline":
                    name = manifest["name"] # name of Thorium pipeline
                    version = manifest.get('version', "latest") # version or if not present latest
                    if (name not in pipelines):
                        pipelines[name] = {}
                    # populate pipeline manifest fields
                    pipelines[name][version] = pipeline_fields(manifest)
                # import this image to the toolbox manifest
                if manifest["type"] == "image":
                    name = manifest["name"] # name of Thorium image
                    version = manifest.get('version', "latest") # version or if not present latest
                    if (name not in images):
                        images[name] = {}
                    # populate image manifest fields
                    images[name][version] = image_fields(manifest)

# Add Pipelines and Images to the market manifest          
toolbox_manifest["pipelines"] = pipelines
toolbox_manifest["images"] = images

with open('toolbox.json', 'w') as toolbox_json_file:
    json.dump(toolbox_manifest, toolbox_json_file, indent=2)
# toml dumping looks really bad here, keys are dumped level by level so different images configs are interspersed
#with open('toolbox.toml', 'w') as toolbox_toml_file:
#    toml.dump(toolbox_manifest, toolbox_toml_file)