from math import exp, log
from itertools import chain
from collections import Counter, defaultdict


class LM:
    def train(self, train_corpus):
        pass

    def perplexity(self, test_corpus):
        pass


def ngrams(seq, n=2):
    return list(zip(*[seq[i:] for i in range(n)]))


def reduced_bigrams(seq):
    from iobes import extract_function, extract_type, TokenFunction

    bigrams = []
    for prev, curr in zip(seq, seq[1:]):
        prev_func = extract_function(prev)
        prev_type = extract_type(prev)
        curr_func = extract_function(curr)
        curr_type = extract_type(curr)
        if prev_func == TokenFunction.BEGIN and curr_func == TokenFunction.INSIDE and prev_type == curr_type:
            bigrams.append((prev, curr))
        elif prev_func == TokenFunction.BEGIN and curr_func == TokenFunction.END and prev_type == curr_type:
            bigrams.append((prev, curr))
        elif prev_func == TokenFunction.INSIDE and curr_func == TokenFunction.INSIDE and prev_type == curr_type:
            bigrams.append((prev, curr))
        elif prev_func == TokenFunction.INSIDE and curr_func == TokenFunction.END and prev_type == curr_type:
            bigrams.append((prev, curr))
    return bigrams


class LaplaceUnigramLM(LM):
    def train(self, train_corpus):
        self.uni = Counter(chain(*train_corpus))
        self.vocab_size = len(self.uni)
        self.uni_sum = sum(self.uni.values())

    def generate_probability(self, word):
        return (self.uni[word] + 1) / (self.uni_sum + self.vocab_size)

    def _log_probability_of_sequence(self, sequence):
        """Calculate the probability of a sequence using the chain rule of probability and a bigram markov assumption.

        This calculation is done in log space to prevent underflow.
        """
        sequence_score = 0
        for word in sequence:
            sequence_score += log(self.generate_probability(word))
        return sequence_score

    def probability_of_sequence(self, sequence):
        """Convert the log probability of a sequence to the normal probability."""
        return exp(self._log_probability_of_sequence(sequence))

    def perplexity(self, test_corpus):
        lp = sum(self._log_probability_of_sequence(seq) for seq in test_corpus)
        length = sum(len(seq) - 2 for seq in test_corpus)
        return exp(-lp / length)


class LaplaceLM(LM):
    def train(self, train_corpus):
        self.uni = Counter(chain(*train_corpus))
        self.bi = Counter(chain(*[ngrams(x) for x in train_corpus]))
        self.vocab_size = len(self.uni)
        self.uni_sum = sum(self.uni.values())

    def generate_probability(self, word, context=None):
        """Generate the probability with LaPlace smoothing."""
        if context is None:
            return (self.uni[word] + 1) / (self.uni_sum + self.vocab_size)
        else:
            return (self.bi[(context, word)] + 1) / (self.uni[context] + self.vocab_size)

    def _log_probability_of_sequence(self, sequence):
        """Calculate the probability of a sequence using the chain rule of probability and a bigram markov assumption.

        This calculation is done in log space to prevent underflow.
        """
        sequence_score = 0
        context = None
        for word in sequence:
            sequence_score += log(self.generate_probability(word, context))
            context = word
        return sequence_score

    def probability_of_sequence(self, sequence):
        """Convert the log probability of a sequence to the normal probability."""
        return exp(self._log_probability_of_sequence(sequence))

    def perplexity(self, test_corpus):
        lp = sum(self._log_probability_of_sequence(seq) for seq in test_corpus)
        length = sum(len(seq) - 2 for seq in test_corpus)
        return exp(-lp / length)


class ReducedLaplaceLM(LaplaceLM):
    def train(self, train_corpus):
        self.uni = Counter(chain(*train_corpus))
        self.bi = Counter(chain(*[reduced_bigrams(x) for x in train_corpus]))
        self.vocab_size = len(self.uni)
        self.uni_sum = sum(self.uni.values())


class KneserNeyLM(LM):
    def __init__(self, d=0.75):
        super().__init__()
        self.d = d

    def train(self, train_corpus):
        self.unigram = Counter(chain(*train_corpus))
        self.bigram = Counter(chain(*[ngrams(x) for x in train_corpus]))
        # Collect all the types that follow each word and precede each word.
        head_pairs = defaultdict(set)
        tail_pairs = defaultdict(set)
        for head, tail in self.bigram:
            head_pairs[head].add(tail)
            tail_pairs[tail].add(head)
        # Convert from sets into counts (with a default of 0)
        # Head count is the number of word types where w appears as the first word in the bigram
        # This gives us `|{w: c(w_{i-1}, w) > 0}|`
        self.head_counts = defaultdict(int, {k: len(v) for k, v in head_pairs.items()})
        # Tail count is the number of word types where w appears as the second word in the bigram
        # This gives us `|{w_{i-1} : c(w_{i-1}, w_i) > 0}|`
        self.tail_counts = defaultdict(int, {k: len(v) for k, v in tail_pairs.items()})

    def prob(self, word, context=None):
        p_continue = self.tail_counts[word] / len(self.bigram)
        p_discount = self.d * self.head_counts[context] / self.unigram[context]
        p = max(self.bigram[(context, word)] - self.d, 0) / self.unigram[context]
        return p + p_discount * p_continue

    def _log_prob_seq(self, seq):
        seq = [x if x in self.unigram else "<unk>" for x in seq]
        return sum(log(self.prob(c, p)) for p, c in ngrams(seq))

    def prob_seq(self, seq):
        return exp(self._log_prob_seq(seq))

    def perplexity(self, test_corpus):
        lp = sum(self._log_prob_seq(seq) for seq in test_corpus)
        length = sum(len(seq) - 2 for seq in test_corpus)
        return exp(-lp / length)


class ReducedKneserNeyLM(KneserNeyLM):
    def train(self, train_corpus):
        self.unigram = Counter(chain(*train_corpus))
        self.bigram = Counter(chain(*[reduced_bigrams(x) for x in train_corpus]))
        # Collect all the types that follow each word and precede each word.
        head_pairs = defaultdict(set)
        tail_pairs = defaultdict(set)
        for head, tail in self.bigram:
            head_pairs[head].add(tail)
            tail_pairs[tail].add(head)
        # Convert from sets into counts (with a default of 0)
        # Head count is the number of word types where w appears as the first word in the bigram
        # This gives us `|{w: c(w_{i-1}, w) > 0}|`
        self.head_counts = defaultdict(int, {k: len(v) for k, v in head_pairs.items()})
        # Tail count is the number of word types where w appears as the second word in the bigram
        # This gives us `|{w_{i-1} : c(w_{i-1}, w_i) > 0}|`
        self.tail_counts = defaultdict(int, {k: len(v) for k, v in tail_pairs.items()})


if __name__ == "__main__":

    def read_data(file_name):
        with open(file_name, "r") as f:
            corpus = f.read()
        return [["<s>"] + x.split() + ["</s>"] for x in corpus.split("\n") if x != ""]

    import sys

    train = read_data(sys.argv[1])
    test = read_data(sys.argv[2])

    lm = KneserNeyLM()
    lm.train(train)
    print(lm.perplexity(test))

    lm = LaplaceLM()
    lm.train(train)
    print(lm.perplexity(test))

    lm = LaplaceUnigramLM()
    lm.train(train)
    print(lm.perplexity(test))
