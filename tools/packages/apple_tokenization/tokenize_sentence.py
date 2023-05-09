import os
import ctypes
import json
import fire

file_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = f"{file_dir}/.build/release/libTokenizer.dylib"
lib = ctypes.CDLL(lib_path)

# Define the C-compatible Swift function signature
tokenize_func = lib.tokenizeSentenceWrapper
tokenize_func.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
tokenize_func.restype = None


def tokenize_sentence(sentence: str) -> list:
    input_string = sentence.encode('utf-8')
    output_buffer = ctypes.create_string_buffer(65536)
    tokenize_func(input_string, output_buffer, len(output_buffer))
    json_string = output_buffer.value.decode('utf-8')
    parsed_result = json.loads(json_string)
    tokenized_sentence = [
        (token, int(start), int(end)) for token, start, end in parsed_result
    ]
    return tokenized_sentence


if __name__ == "__main__":
    fire.Fire(tokenize_sentence)
