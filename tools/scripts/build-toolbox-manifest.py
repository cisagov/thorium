import tomllib #tomllib from std only supports load/loads, toml supports dump but has awful formatting
import json
import os
import logging
import argparse

def image_fields(manifest, root, toolbox, override_path):
    """Build an image's toolbox fields from an image manifest"""
    image = {}
    # Registry image name
    name = manifest.get("name", "")
    image_name = manifest.get("image_name", "")

    image_tags = []
    image_version = manifest.get("version", "")
    build_image = manifest.get("build", True)
    base_image_token = manifest.get("base_image_token")
    allow_base_override = manifest.get("allow_base_override", True)
    if (image_version == ""):
        logging.error(f" No image version found for {name}")
    if (image_name == ""):
        logging.error(f" No image name found for {name}")
    else:
        registries = toolbox.get("registries", [])
        if "registry" in toolbox and toolbox.get("registry") not in registries:
            registries.append(toolbox.get("registry"))
        for registry in registries:
            tag = ""
            # use tool name rather than image name field to build container path
            if override_path:
                tag = f"{registry}/{name}:{image_version}"
            # use manifest image_name field for full container path
            else:
                tag = f"{registry}/{image_name}:{image_version}"
            # only add image if not duplicated by legacy "registry" key
            if tag not in image_tags:
                image_tags.append(tag)

    # Container build context path
    image_build_path = f"{root}"
    if (manifest["build_path"] not in ['./', '.']):
        image_build_path += f"/{manifest["build_path"]}"
    image["build_path"] = image_build_path
    image["build_image"] = build_image
    image["image_tags"] = image_tags
    image["allow_base_override"] = allow_base_override
    if base_image_token:
        image["base_image_token"] = base_image_token
    # Thorium image configuration
    if ('config_from' in manifest and 'config' not in manifest):
        config_path = f"{root}/{manifest['config_from']}"
        with open(config_path, 'rb') as config_file:
            config = json.load(config_file)
            if len(image_tags) == 0:
                logging.error(f"No image tag specified, leaving {name} image config blank")
            else:
                config["image"] = image_tags[0]
            image["config"] = config
    return image

def pipeline_fields(manifest, root):
    """Build an pipelines's toolbox fields from an pipeline manifest"""
    pipeline = {}
    pipeline["description"] = manifest.get("description", "")
    pipeline["images"] = manifest.get("images", {})
    if ('config_from' in manifest and 'config' not in manifest):
        config_path = f"{root}/{manifest['config_from']}"
        with open(config_path, 'rb') as config_file:
            config = json.load(config_file)
            pipeline["config"] = config
    return pipeline

def main():
    """Build a toolbox manifest from a project containing images and pipelines"""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config",
        help="Toolbox config (JSON/TOML)",
        type=str,
        required=True,
        )
    argparser.add_argument("-o", "--override_path",
        help="Use manifest image_name instead of tool name in container tag",
        action='store_true',
        )
    # create empty market place manifest
    toolbox = {
        "pipelines": {},
        "images": {},
    }
    try:
        # load toolbox manifest file
        args = argparser.parse_args() 
        toolbox_config_path = args.config
        with open(toolbox_config_path, 'rb') as config_file:
            config = tomllib.load(config_file)
        toolbox["name"] = config.get("name", "")
        toolbox["registry"] = config.get("registry", "")
        toolbox["registries"] = config.get("registries", [])
        image_path_prefix = config.get("override_image_path_prefix", "")

    except FileNotFoundError:
        logging.error(f"Failed to find toolbox manifest {toolbox_config_path}")
        exit(1)
    except tomllib.TOMLDecodeError as e:
        logging.error(f"Failed to decode TOML toolbox config ${toolbox_config_path} with error: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Failed to load toolbox config with unexpected error: {e}")
        exit(1) 
    
    images = dict()
    pipelines = dict()
    # loop through any directories looking for manifests
    for root, _, files in os.walk('.'):
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
                        pipelines[name][version] = pipeline_fields(manifest, root)
                    # import this image to the toolbox manifest
                    if manifest["type"] == "image":
                        name = manifest["name"] # name of Thorium image
                        version = manifest.get('version', "latest") # version or if not present latest
                        if (name not in images):
                            images[name] = {}
                        # populate image manifest fields
                        images[name][version] = image_fields(manifest, root, toolbox, args.override_path)
    
    # Add Pipelines and Images to the market manifest          
    toolbox["pipelines"] = pipelines
    toolbox["images"] = images
    
    with open('toolbox.json', 'w') as toolbox_json_file:
        json.dump(toolbox, toolbox_json_file, indent=2)
    # toml dumping looks really bad here, keys are dumped level by level so different images configs are interspersed
    #with open('toolbox.toml', 'w') as toolbox_toml_file:
    #    toml.dump(toolbox_manifest, toolbox_toml_file)


if __name__ == "__main__":
    main()

