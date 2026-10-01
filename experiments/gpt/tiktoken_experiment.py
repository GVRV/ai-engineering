import tiktoken
enc = tiktoken.get_encoding('cl100k_base')

def compare_tokenizer_with_characters(text):
    num_charaters = len(text)
    print(f"Number of characters: {num_charaters}")

    tokens = enc.encode(text)
    num_tokens = len(tokens)
    print(f"Number of tokens: {num_tokens}")

    compression = float(num_charaters) / num_tokens
    print(f"Compression: {compression:.2f}")

compare_tokenizer_with_characters("To be, or not to be, that is the question:")

TEXT = open('input.txt', 'r', encoding='utf-8').read()
compare_tokenizer_with_characters(TEXT[:2000])