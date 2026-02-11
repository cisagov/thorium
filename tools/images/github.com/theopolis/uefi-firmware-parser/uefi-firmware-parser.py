#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Based on https://github.com/theopolis/uefi-firmware-parser/blob/master/bin/uefi-firmware-parser
#
# Copyright (c) 2014 Teddy Reed <teddy@prosauce.org>
# Copyright (c) 2013 Hector Martin <hector@marcansoft.com>
#
# MIT License

import argparse
import json
import logging
import os
import pathlib
import re
import sys

from uefi_firmware.uefi import *
from uefi_firmware.generator import uefi as uefi_generator
from uefi_firmware import AutoParser
import uefi_firmware.utils # import nocolor

def _find_objects(_object, types):
    """
    Recursively search for objects with the given types below the given object.
    """
    objects = []
    for _object2 in _object.objects:
        if _object2 is None:
            continue

        _type = None
        if hasattr(_object2, "type"):
            _type = EFI_SECTION_TYPES.get(_object2.type)

        if _type is not None and _type[2] in types:
            objects.append(_object2)

        objects.extend(_find_objects(_object2, types))

    return objects


def _sanitize_filename(s):
    """
    Sanitize weird characters from the filename and replace with '_'.

    TODO: should double check this
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', s)
    return sanitized.strip().strip(".")


def _extract_objects(parsed_object):
    """
    Recursively extract objects that appear to be executables
    """
    count = 0
    for _object in parsed_object.objects:
        if _object is None:
            continue

        if isinstance(_object, FirmwareFile):
            if EFI_FILE_TYPES[_object.type][2] in ["RAW"]:
                count += _extract_objects(_object)
                continue
            elif EFI_FILE_TYPES[_object.type][2] in ["FV_IMAGE"]:
                count += _extract_objects(_object)
                continue

            bins = _find_objects(_object, {"PE32", "PIC", "TE"})
            names = _find_objects(_object, {"UI"})
            versions = _find_objects(_object, {"VERSION"})

            if len(bins) == 0:
                continue

            if len(bins) > 1 or len(names) > 1:
                logging.warning("more than one binary and name found, using only the first of each")

            name = None
            if len(names) == 1:
                name = _sanitize_filename(names[0].name)
            else:
                name = _sanitize_filename(uefi_firmware.utils.sguid(_object.guid))

            fpath = pathlib.Path(args.output).joinpath(name + ".bin")

            i = 0

            while fpath.exists():
                i += 1
                fpath = fpath.with_name(name + f"-{i}.bin")

            logging.info("writing to binary: %s", fpath)
            with fpath.open("wb") as f:
                f.write(bins[0].data)

            count += 1

        # recurse recurse
        count += _extract_objects(_object)

    return count


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Parse, and optionally output, details and data on UEFI-related firmware.")
    argparser.add_argument(
        "--color", default="auto", choices=("always", "never", "auto"),
        help="Control the use of ANSI colors in the output. (auto is default)")
    argparser.add_argument(
        '-p', "--nocolor", const="never", dest="color", action="store_const",
        help="Plain text output. Do not use ANSI colors. (Alias for --color=never)")
    argparser.add_argument(
        '-o', "--output", default=".",
        help="Dump firmware objects to this folder.")
    argparser.add_argument(
        '-O', "--outputfolder", default=False, action="store_true",
        help="Dump firmware objects to a folder based on filename ${FILENAME}_output/ ")
    argparser.add_argument(
        '-e', "--extract", action="store_true",
        help="Extract all files/sections/volumes.")
    argparser.add_argument(
        '-s', "--structure",
        type=argparse.FileType("w"),
        default=None,
        help="Write structure to file (default: stdout).")
    argparser.add_argument(
        '--verbose', default=False, action='store_true',
        help='Enable verbose logging while parsing')
    argparser.add_argument(
        "file",
        help="The file to work on")
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    # Do not use colors when piping the output
    if args.color == "auto":
        args.color = "always" if sys.stdout.isatty() else "never"
    # Pass the color flag to the util config
    uefi_firmware.utils.nocolor = (args.color == "never")

    try:
        with open(args.file, 'rb') as fh:
            input_data = fh.read()
    except Exception as e:
        logging.exception("cannot read file (%s).", args.file)
        sys.exit(1)

    parser = AutoParser(input_data, search=True)

    if parser.type() == 'unknown':
        logging.error("cannot parse %s, could not detect firmware type.", args.file)
        sys.exit(2)

    firmware = parser.parse()

    if args.structure:
        res = firmware.to_dict()
        json.dump(res, args.structure)

    if args.outputfolder:
        autodir = f"{args.file}_output"
        args.output = autodir
        logging.info("writing to autodir: %s", autodir)

    os.makedirs(args.output, exist_ok=True)

    count = _extract_objects(firmware)
    print(f"extracted {count} binaries")

    if args.structure:
        args.structure.close()
