from src.models.ngram_model import NGramModel
import numpy as np


class ModelEvaluate:

    @staticmethod
    def get_oov_count(train_vocabs, test_vocab):
        oov = 0
        hum_train_vocab, gpt_train_vocab = train_vocabs
        for i in test_vocab:
            if i not in hum_train_vocab and i not in gpt_train_vocab:
                oov += test_vocab[i]
        return oov

    @staticmethod
    def calc_oov_rate(vocabs, total_oov_count):
        total_vocabs = 0
        for vocab in vocabs:
            total_vocabs += sum(vocab.values())

        return total_oov_count / total_vocabs

    @staticmethod
    def calculate_log_prob(class_k_train_gram, class_k_1_train_gram, k, class_prob, doc, vocab_size):
        """Calculates the log probability for a given document."""
        ngram_model = NGramModel(k)
        ngrams = ngram_model.get_ngrams([doc])
        log_prob = np.log(class_prob) + sum(
            np.log((class_k_train_gram.get(ngram, 0) + 1) /
                   (class_k_1_train_gram.get((ngram[0],), 0) + vocab_size))
            for ngram in ngrams.freq_table
        )
        return log_prob

    @staticmethod
    def get_class_prob(lst_traindocs):
        prob_class_1 = len(lst_traindocs[0]) / (len(lst_traindocs[0]) + len(lst_traindocs[1]))
        prob_class_2 = 1 - prob_class_1

        return prob_class_1, prob_class_2

    @staticmethod
    def calculate_accuracy(train_docs, test_docs, k_train_grams, vocab_size, k):
        """Calculates model accuracy for n-grams."""

        hum_trainset, gpt_trainset = train_docs
        hum_testset, gpt_testset = test_docs
        human_k_train_gram, gpt_k_train_gram = k_train_grams

        P_Y_Human, P_Y_GPT = ModelEvaluate.get_class_prob([hum_trainset, gpt_trainset])

        # Calculate n-1 grams
        hum_train_nvocab = NGramModel(k - 1)
        hum_train_nvocab.get_ngrams(hum_trainset)

        gpt_train_nvocab = NGramModel(k - 1)
        gpt_train_nvocab.get_ngrams(gpt_trainset)

        correct_human = 0
        total_human = 0
        for doc in hum_testset:
            total_human += 1
            true_label = "Human"
            log_hum_prob = ModelEvaluate.calculate_log_prob(human_k_train_gram.freq_table, hum_train_nvocab.freq_table,
                                                            k, P_Y_Human, doc, vocab_size)
            log_gpt_prob = ModelEvaluate.calculate_log_prob(gpt_k_train_gram.freq_table, gpt_train_nvocab.freq_table, k,
                                                            P_Y_GPT, doc, vocab_size)
            predicted_label = "Human" if log_hum_prob > log_gpt_prob else "GPT"
            if predicted_label == true_label:
                correct_human += 1

        correct_gpt = 0
        total_gpt = 0
        for doc in gpt_testset:
            total_gpt += 1
            true_label = "GPT"
            log_hum_prob = ModelEvaluate.calculate_log_prob(human_k_train_gram.freq_table, hum_train_nvocab.freq_table,
                                                            k, P_Y_Human, doc, vocab_size)
            log_gpt_prob = ModelEvaluate.calculate_log_prob(gpt_k_train_gram.freq_table, gpt_train_nvocab.freq_table, k,
                                                            P_Y_GPT, doc, vocab_size)
            predicted_label = "GPT" if log_gpt_prob > log_hum_prob else "Human"
            if predicted_label == true_label:
                correct_gpt += 1

        return (correct_human + correct_gpt) / (total_human + total_gpt)
