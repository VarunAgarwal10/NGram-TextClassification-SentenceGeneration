from src.models.ngram_model import NGramModel
from src.utils.file_handler import FileManager, DocumentProcessor, TrainTestGenerator
from src.utils.model_evaluate import ModelEvaluate


class Runner:
    """Orchestrates the entire workflow."""

    def __init__(self, ai_file, human_file):
        self.ai_file = ai_file
        self.human_file = human_file

    def run(self):
        """Runs the entire workflow."""
        ai_docs = FileManager.read_file(self.ai_file)
        hum_docs = FileManager.read_file(self.human_file)

        # Preprocessing
        ai_cleaned = DocumentProcessor.preprocess_docs(ai_docs)
        human_cleaned = DocumentProcessor.preprocess_docs(hum_docs)

        # Data splitting
        ai_train, ai_test = TrainTestGenerator.split_docs(ai_cleaned, train_ratio=0.9)
        human_train, human_test = TrainTestGenerator.split_docs(human_cleaned, train_ratio=0.9)

        # Get the ngram vocabs for both the train and test sets
        # Instantiating Bigram models
        hum_train_bivocab = NGramModel(2).get_ngrams(human_train)
        hum_test_bivocab = NGramModel(2).get_ngrams(human_test)
        ai_train_bivocab = NGramModel(2).get_ngrams(ai_train)
        ai_test_bivocab = NGramModel(2).get_ngrams(ai_test)

        # Instantiating Trigram models
        hum_train_trivocab = NGramModel(3).get_ngrams(human_train)
        hum_test_trivocab = NGramModel(3).get_ngrams(human_test)
        ai_train_trivocab = NGramModel(3).get_ngrams(ai_train)
        ai_test_trivocab = NGramModel(3).get_ngrams(ai_test)

        # Bigram Evaluation: OOV Rate
        bigram_hum_oov = ModelEvaluate.get_oov_count(
            [hum_train_bivocab.freq_table, ai_train_bivocab.freq_table],
            hum_test_bivocab.freq_table
        )
        bigram_ai_oov = ModelEvaluate.get_oov_count(
            [hum_train_bivocab.freq_table, ai_train_bivocab.freq_table],
            ai_test_bivocab.freq_table
        )

        total_bigram_oov = bigram_hum_oov + bigram_ai_oov
        bigram_oov_rate = ModelEvaluate.calc_oov_rate(
            [hum_test_bivocab.freq_table, ai_test_bivocab.freq_table],
            total_bigram_oov
        )
        print(f"Bigram OOV Rate: {round(bigram_oov_rate * 100, 3)}%")

        # Trigram Evaluation: OOV Rate
        trigram_hum_oov = ModelEvaluate.get_oov_count(
            [hum_train_trivocab.freq_table, ai_train_trivocab.freq_table],
            hum_test_trivocab.freq_table
        )
        trigram_ai_oov = ModelEvaluate.get_oov_count(
            [hum_train_trivocab.freq_table, ai_train_trivocab.freq_table],
            ai_test_trivocab.freq_table
        )

        total_trigram_oov = trigram_hum_oov + trigram_ai_oov
        trigram_oov_rate = ModelEvaluate.calc_oov_rate(
            [hum_test_trivocab.freq_table, ai_test_trivocab.freq_table],
            total_trigram_oov
        )
        print(f"Trigram OOV Rate: {round(trigram_oov_rate * 100, 3)}%")

        # Getting the total AI and Human vocab (train + test) for Bigram
        bigram_vocab_train = NGramModel.merge_vocab(hum_train_bivocab.freq_table, ai_train_bivocab.freq_table)
        bigram_vocab_test = NGramModel.merge_vocab(hum_test_bivocab.freq_table, ai_test_bivocab.freq_table)

        # Total Bigram Vocabulary for AI and Human Corpus
        total_bivocab = NGramModel.merge_vocab(bigram_vocab_train, bigram_vocab_test)

        # Getting the total AI and Human vocab (train + test) for Trigram
        trigram_vocab_train = NGramModel.merge_vocab(hum_train_trivocab.freq_table, ai_train_trivocab.freq_table)
        trigram_vocab_test = NGramModel.merge_vocab(hum_test_trivocab.freq_table, ai_test_trivocab.freq_table)

        # Total Trigram Vocabulary for AI and Human Corpus
        total_trivocab = NGramModel.merge_vocab(trigram_vocab_train, trigram_vocab_test)

        # Total Vocabulary Size: |V|
        print(f"\nTotal Bigram Vocabulary Size: {len(total_bivocab)}")
        print(f"Total Trigram Vocabulary Size: {len(total_trivocab)}")

        # Getting the Bigram Model accuracy
        bigram_model_accuracy = ModelEvaluate.calculate_accuracy(
            train_docs=[human_train, ai_train],
            test_docs=[human_test, ai_test],
            k_train_grams=[hum_train_bivocab, ai_train_bivocab],
            vocab_size=len(total_bivocab),
            k=2
        )
        print(f"\nBigram Model Accuracy: {round(bigram_model_accuracy * 100, 3)}%")

        # Getting the Trigram Model Accuracy
        trigram_model_accuracy = ModelEvaluate.calculate_accuracy(
            train_docs=[human_train, ai_train],
            test_docs=[human_test, ai_test],
            k_train_grams=[hum_train_trivocab, ai_train_trivocab],
            vocab_size=len(total_trivocab),
            k=3
        )
        print(f"Trigram Model Accuracy: {round(trigram_model_accuracy * 100, 3)}%")


if __name__ == '__main__':
    AI_FILE = "../data/ai_generated_docs.txt"
    HUMAN_FILE = "../data/human_generated_docs.txt"
    workflow = Runner(AI_FILE, HUMAN_FILE)
    workflow.run()
