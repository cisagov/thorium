#!/usr/bin/env python3

import io
import sys
import json

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import PDFStream
from pdfminer.pdfexceptions import PDFObjectNotFound
from pathlib import Path

def mine_pdf(pdf_buff, children):
    # parse our pdf
    parser = PDFParser(pdf_buff)
    # convert our parsed pdf into a document
    pdf = PDFDocument(parser)
    # build a map of our tool results
    results = {'capabilities': [], "xrefs": {}}
    # step over the xrefs in this pdf
    for xref in pdf.xrefs:
        # get the object ids in this xref
        for id in xref.get_objids():
            # get our js object
            try:
                obj = pdf.getobj(id)
            except PDFObjectNotFound:
                print("Warning: Missing object {id}!")
            # add this objec to our xref map
            results["xrefs"][id] = obj
            # handle objects that are dictonaries
            if isinstance(obj, dict):
                # check if this object has a javascript key
                if 'JS' in obj:
                    # add a capability tag for javascript if it doesn't already exist
                    if "EmbeddedJavascript" not in results["capabilities"]:
                        results["capabilities"].append("EmbeddedJavascript")
                    # build the path to write this javascript stream too
                    child_path = children.joinpath(f"pdf_js_stream_{id}")
                    # write this pdf stream off to disk
                    with open(child_path, 'wb') as fp:
                        print(f"Writting javascript stream {id} to disk")
                        # write this javascript off to disk
                        fp.write(obj['JS'])
            # handle pdf streams
            if isinstance(obj, PDFStream):
                # add a capability tag for javascript if it doesn't already exist
                if "EmbeddedFile" not in results["capabilities"]:
                    results["capabilities"].append("EmbeddedFile")
                # get this streams data
                child_data = obj.get_data()
                # build the path to write this child data too
                child_path = children.joinpath(f"pdf_stream_{id}")
                # write this pdf stream off to disk
                with open(child_path, 'wb') as fp:
                    print(f"Writting stream {id} to disk")
                    fp.write(child_data)
    # return our xrefs
    return results
    

if __name__ == "__main__":
    # try to load our file
    with open(sys.argv[1], 'rb') as fp:
        # read ouf pdf
        pdf_buff = io.BytesIO(fp.read())
        # mine our pdf for javascript
        results = mine_pdf(pdf_buff, Path("/tmp/thorium/children/carved/unknown"))
        # write our results off to disk
        with open("/tmp/thorium/results", 'w') as fp:
            # the default serailize is probably not production ready
            json.dump(results, fp, default=str)
