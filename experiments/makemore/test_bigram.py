from bigram import get_bigram_dict, get_bigram_counts_tensor, make_more, get_bigram_probabilistic_tensor, calculate_nll_of_dataset


def test_dataset_sanity():
    words = open('names.txt', 'r').read().splitlines()
    assert len(words) == 32033
    assert min(len(w) for w in words) == 2
    assert max(len(w) for w in words) == 15

def test_bigram_dict():
    bigram = get_bigram_dict(['emma'])
    assert len(bigram.keys()) == 5
    assert bigram[('<S>', 'e')] == 1
    assert bigram[('e', 'm')] == 1
    assert bigram[('m', 'm')] == 1
    assert bigram[('m', 'a')] == 1
    assert bigram[('a', '<E>')] == 1

    words = open('names.txt', 'r').read().splitlines()
    bigram = get_bigram_dict(words)

    # No empty words
    assert ('<S>', '<E>') not in bigram


def test_bigram_counts_tensor():
    bigram = get_bigram_counts_tensor()

    # No empty words
    assert bigram[0][0].item() == 0


def test_bigram_probabilities_tensor():
    bigram = get_bigram_probabilistic_tensor(get_bigram_counts_tensor())

    # Very small probability because of smoothing out
    assert bigram[0][0].item() != 0

    # Broading bug sanity check
    assert bigram[0].sum().item() == 1.0

    make_more(bigram)


def test_normalised_negative_log_likelihood_for_dataset():
    calculate_nll_of_dataset()
