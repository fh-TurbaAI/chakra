import argparse
import gzip

import orjson
from google.protobuf.json_format import ParseDict
from tqdm import tqdm

from ...schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from ..third_party.utils.protolib import encodeMessage as encode_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts Chakra execution trace in JSON format to ET format.")
    parser.add_argument(
        "--input_filename",
        type=str,
        required=True,
        help="Specifies the input filename of the jsonized Chakra execution trace.",
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Specifies the output filename for the ET data."
    )
    args = parser.parse_args()

    with (
        gzip.open(args.input_filename, "rb")
        if args.input_filename.endswith(".gz")
        else open(args.input_filename, "r") as file_in,
        gzip.open(args.output_filename, "w")
        if args.output_filename.endswith(".gz")
        else open(args.output_filename, "wb") as file_out
    ):
        trace_objects = orjson.loads(file_in.read())
        global_metadata = ParseDict(trace_objects[0], GlobalMetadata())
        encode_message(file_out, global_metadata)
        for sub_dict in tqdm(trace_objects[1:], desc="Saving chakra nodes", unit="node"):
            encode_message(file_out, ParseDict(sub_dict, ChakraNode()))

if __name__ == "__main__":
    main()
