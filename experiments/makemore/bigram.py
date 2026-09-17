import torch

WORDS = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(WORDS))))
num_chars = len(chars) + 1 # special '.' character

ch_to_i = {s: i+1 for i,s in enumerate(chars)}
ch_to_i['.'] = 0

i_to_ch = {i+1: s for i,s in enumerate(chars)}
i_to_ch[0] = '.'


def get_generator():
    return torch.Generator().manual_seed(2147483647)


def get_bigram_dict(words):
    if words is None:
        words = WORDS

    bigram = {}

    for w in words:
        characters = ['<S>'] + list(w) + ['<E>']
        for ch1, ch2 in zip(characters, characters[1:]):
            ch_pair = (ch1, ch2)
            bigram[ch_pair] = bigram.get(ch_pair, 0) + 1

    return bigram


def get_bigram_counts_tensor():
    bigram = torch.zeros((num_chars, num_chars), dtype=torch.int32)

    for w in WORDS:
        characters = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(characters, characters[1:]):
            ix1 = ch_to_i[ch1]
            ix2 = ch_to_i[ch2]

            bigram[ix1][ix2] += 1

    return bigram


def get_bigram_probabilistic_tensor(bigram_counts):
    # Smooth out the probabilities by making sure there is no zero probabilities
    # to ensure we don't get -INF when calculating the log likelihood
    bigram_probabilities = (bigram_counts + 1).float()
    bigram_probabilities /= bigram_probabilities.sum(1, keepdims=True)
    return bigram_probabilities


def make_more(bigram):
    g = get_generator()
    log_likelihood = 0.0
    num_predictions = 0

    for _ in range(20):
        ix = 0
        name = ''

        while True:
            probabilities = bigram[ix]
            # COMPARE: Inefficient row-wise probability distribution calculation
            # counts = bigram[ix].float()
            # probabilities = counts / counts.sum()

            # COMPARE: Completely uniform probability distribution
            # probabilities = torch.ones(27) / 27.0

            ix = torch.multinomial(probabilities, num_samples=1, replacement=True, generator=g).item()
            num_predictions += 1

            #
            # Probability will be between [0, 1] - We want to maximise probability of each prediction
            # P(prediction1) * P(prediction2) * P(prediction3) <-- this needs to be as high as possible
            #
            # Log probability will between [-INF, 0]
            # Because probabilities are tiny, and multiplying them results in even tinier numbers
            # P(x) * P(y) * P(z) => log(P(x)) + log(P(y)) + log(P(z))
            # So we now cumulatively collect the log likehoods
            #
            # But because this will be a negative number, we negate it to get a positive value
            # which we can minimise instead of maximising like a tradition loss function
            #
            # We can also normalise this negative log likelihood by dividing the neagtive log likelihood
            # with the number of predictions to get an average loss per prediction
            #
            log_probability = torch.log(probabilities[ix])
            log_likelihood += log_probability

            name += i_to_ch[ix]

            if ix == 0:
                print(name)
                break

    negative_log_likelihood = -log_likelihood
    print(f'{negative_log_likelihood=}')
    normalised_nll = negative_log_likelihood / num_predictions
    print(f'{normalised_nll=}')


def calculate_nll_of_dataset():
    bigram = get_bigram_probabilistic_tensor(get_bigram_counts_tensor())
    log_likelihood = 0
    num_predictions = 0

    for w in WORDS:
        characters = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(characters, characters[1:]):
            ix1 = ch_to_i[ch1]
            ix2 = ch_to_i[ch2]

            prob = bigram[ix1, ix2]
            log_prob = torch.log(prob)
            log_likelihood += log_prob
            num_predictions += 1

    print(f'{log_likelihood=}')
    nll = -log_likelihood
    print(f'{nll=}')
    print(f'{num_predictions=}')
    nnll = nll / num_predictions
    print(f'{nnll=}')

