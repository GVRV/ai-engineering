import regex as re

TEXT = open('input.txt', 'r').read()
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def encode(input_str):
    output_tokens = []
    for split in re.findall(PAT, input_str):
        input_utf8 = split.encode('utf-8')
        output_tokens.append(list(map(int, input_utf8)))
    return output_tokens

input_text = TEXT[:20000]
input_str = encode(input_text)

STARTING_TOKEN = 256
assert [x for split in input_str for x in split if x >= STARTING_TOKEN] == []

def get_stats(tokens):
    counts = {}
    for split in tokens:
        for pair in zip(split, split[1:]):
            counts[pair] = counts.get(pair, 0) + 1

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)

def merge(splits, replace_token_pair, replace_token_id):
    new_splits = []

    for tokens in splits:
        new_tokens = []
        idx = 0

        if len(tokens) == 1:
            new_splits.append(tokens)
            continue

        while idx < len(tokens) - 1:
            pair = (tokens[idx], tokens[idx+1])
            if pair == replace_token_pair:
                new_tokens.append(replace_token_id)
                idx += 2 # Because we just replaced 2 tokens with 1 new one

                # Fix for at the end of the list - 1, if we replace the
                # pair and now only 1 element is remaining, we wont
                # process it because we want to process pairs
                if idx == len(tokens) - 1:
                    new_tokens.append(tokens[idx])
            else:
                new_tokens.append(tokens[idx])
                # Fix for at the end of the list, if we're not
                # replacing a pair, we replace the last one as well
                if idx == len(tokens) - 2:
                    new_tokens.append(tokens[idx+1])
                idx += 1

        new_splits.append(new_tokens)
    return new_splits

# Sanity test
assert (merge([[5, 6, 6, 7, 9, 1]], (6, 7), 99)) == [[5, 6, 99, 9, 1]]
assert (merge([[5, 6, 99, 9, 1]], (9, 1), 100)) == [[5, 6, 99, 100]]

def decode(tokens, merges):
    unmerges = {v: k for k,v in merges.items()}
    while True:
        existing_merges = set(tokens) & set(unmerges.keys())

        # If the list no longer contains any values which
        # we can unmerge, we're done
        if not existing_merges:
            break

        new_tokens = []
        idx = 0
        while idx < len(tokens):
            if tokens[idx] not in unmerges.keys():
                new_tokens.append(tokens[idx])
            else:
                token1, token2 = unmerges[tokens[idx]]
                new_tokens.append(token1)
                new_tokens.append(token2)
            idx += 1
        tokens = new_tokens
    return tokens

# Sanity check
assert decode([4, 1, 3, 4], {(5, 6): 4, (7, 8): 5}) == [7, 8, 6, 1, 3, 7, 8, 6]

# Hyperparameter
# Either fixed value or derived from desired vocabulary size
# compared to current vocabulary size
NUM_MERGES = 300
merges = {}
new_token_id = STARTING_TOKEN

input_tokens = list(input_str)
print(f"Starting length of text: {len([x for split in input_tokens for x in split])}")
for _ in range(NUM_MERGES):
    most_common_token_pair, count = get_stats(input_tokens)[0]

    # Ran out of merges to optimise
    if count == 1:
        break

    # Information/debug
    # replacing_string = "".join(chr(_) for _ in decode(most_common_token_pair, merges))
    # print(f"-->{replacing_string}<--:{count}")

    input_tokens = merge(input_tokens, most_common_token_pair, new_token_id)
    merges[most_common_token_pair] = new_token_id
    new_token_id += 1
    # print(f"New length of text: {len(input_tokens)}")

input_flat_str = [x for split in input_str for x in split]
input_flat_tokens = [x for split in input_tokens for x in split]
print(f"Old text length: {len(input_flat_str)}")
print(f"New text length: {len(input_flat_tokens)}")
compression = float(len(input_flat_str)) / float(len(input_flat_tokens))
print(f"Compression: {compression:.2f}")

decoded_input = decode(input_flat_tokens, merges)
decoded_str = "".join(chr(x) for x in decoded_input)
print(f"Length of decoded input: {len(decoded_input)}")
assert decoded_str == input_text
