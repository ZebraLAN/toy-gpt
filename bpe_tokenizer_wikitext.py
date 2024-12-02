import sys
import traceback
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Tuple

import IPython
from datasets import load_dataset


def interactive_debug_on_exception(func):
    """A decorator that starts an IPython session when an exception occurs, with access to local variables."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Print the formatted stack trace
            print("An exception occurred:")
            traceback.print_exc(file=sys.stdout)  # Pretty print exception information

            # Get the current exception's traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_frame = (
                exc_traceback.tb_next.tb_frame
                if exc_traceback.tb_next
                else exc_traceback.tb_frame
            )

            # Use the frame's local namespace in IPython shell
            if tb_frame is not None:
                local_namespace = tb_frame.f_locals.copy()
                print(
                    "\nEntering interactive debug mode. Variables at the exception site are available."
                )
                IPython.embed(user_ns=local_namespace)  # Start IPython with local vars
            else:
                print("Could not access the traceback frame.")

            raise  # Re-raise the exception after inspection

    return wrapper


# Define the separator token as a global constant
SEP_TOKEN = "[SEP]"
SEP_TOKEN_BYTES = SEP_TOKEN.encode("utf-8")


def byte_pair_encoding(
    texts: List[str], num_merges: int
) -> Tuple[List[int], Dict[Tuple[int, int], int]]:
    # Join texts with `[SEP]` token
    complete_text = SEP_TOKEN_BYTES.join(text.encode("utf-8") for text in texts)

    byte_text = list(complete_text)
    token_id = max(byte_text) + 1
    merge_map = {}

    for _ in range(num_merges):
        pairs = get_byte_pair_frequency(byte_text)
        if not pairs:
            break

        most_frequent_pair = max(pairs, key=pairs.get)

        if most_frequent_pair not in merge_map:
            merge_map[most_frequent_pair] = token_id
            token_id += 1

        byte_text = merge_byte_pair(
            byte_text, most_frequent_pair, merge_map[most_frequent_pair]
        )

    return byte_text, merge_map


def get_byte_pair_frequency(byte_sequence: List[int]) -> Dict[Tuple[int, int], int]:
    pairs = defaultdict(int)
    for i in range(len(byte_sequence) - 1):
        pair = (byte_sequence[i], byte_sequence[i + 1])
        pairs[pair] += 1
    return pairs


def merge_byte_pair(
    byte_sequence: List[int], pair: Tuple[int, int], new_token: int
) -> List[int]:
    new_sequence = []
    i = 0
    while i < len(byte_sequence):
        if (
            i < len(byte_sequence) - 1
            and (byte_sequence[i], byte_sequence[i + 1]) == pair
        ):
            new_sequence.append(new_token)
            i += 2
        else:
            new_sequence.append(byte_sequence[i])
            i += 1
    return new_sequence


def decode_byte_pair(
    encoded_sequence: List[int], merge_map: Dict[Tuple[int, int], int]
) -> str:
    reverse_merge_map = {v: k for k, v in merge_map.items()}
    stack = encoded_sequence[::-1]
    decoded_sequence = []

    while stack:
        token = stack.pop()
        if token in reverse_merge_map:
            first, second = reverse_merge_map[token]
            stack.append(second)
            stack.append(first)
        else:
            decoded_sequence.append(token)

    return bytes(decoded_sequence).decode("utf-8")


def split_on_sep(decoded_text: str, sep_token: str = SEP_TOKEN) -> List[str]:
    return decoded_text.split(sep_token)


# Load the Dataset
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
texts = dataset["text"][:100]  # Using a subset for this example

# Tokenize using BPE
num_merges = 5000
encoded, merge_map = byte_pair_encoding(texts, num_merges)

decoded = decode_byte_pair(encoded, merge_map)
text_blocks = split_on_sep(decoded)

for i, (original, block) in enumerate(zip(texts, text_blocks)):
    print(f"Original Block {i}: {original}")
    print(f"Decoded Block {i}: {block}")
    print("---")
