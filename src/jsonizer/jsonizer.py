import argparse
import gzip

import orjson
from google.protobuf.json_format import MessageToDict
from tqdm import tqdm

from ...schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
)
from ...schema.protobuf.et_def_pb2 import (
    Node as ChakraNode,
)
from ..third_party.utils.protolib import decodeMessage as decode_message
from ..third_party.utils.protolib import openFileRd as open_file_rd


def main() -> None:
    parser = argparse.ArgumentParser(description="Converts Chakra execution trace to JSON format.")
    parser.add_argument(
        "--input_filename", type=str, required=True, help="Specifies the input filename of the Chakra execution trace."
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Specifies the output filename for the JSON data."
    )
    args = parser.parse_args()

    execution_trace = open_file_rd(args.input_filename)
    node = ChakraNode()
    trace_objects: list = []
    global_metadata = GlobalMetadata()
    decode_message(execution_trace, global_metadata)
    trace_objects.append(MessageToDict(global_metadata))
    progress_bar = tqdm(desc="Loading chakra nodes", unit="node")
    while decode_message(execution_trace, node):
        trace_objects.append(MessageToDict(node))
        progress_bar.update(1)
    progress_bar.close()
    with (
        gzip.open(args.output_filename, "wb")
        if args.output_filename.endswith(".gz")
        else open(args.output_filename, "wb") as file
    ):
        file.write(orjson.dumps(trace_objects, option=orjson.OPT_INDENT_2))

    execution_trace.close()


if __name__ == "__main__":
    main()
