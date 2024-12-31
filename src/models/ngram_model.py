class NGramModel:

    def __init__(self, k):
        self.k = k
        self.freq_table = {}

    def get_ngrams(self, docs):
        for i in range(len(docs)):
            split_text = docs[i].split()
            for j in range(len(split_text) - self.k + 1):
                ngram = tuple(split_text[j:j + self.k])
                if ngram in self.freq_table:
                    self.freq_table[ngram] += 1
                else:
                    self.freq_table[ngram] = 1
        return self

    @staticmethod
    def merge_vocab(vocab1, vocab2):
        merged_dict = vocab1.copy()
        for ngram in vocab2:
            if ngram in merged_dict:
                merged_dict[ngram] += vocab2[ngram]
            else:
                merged_dict[ngram] = vocab2[ngram]
        return merged_dict
