import random

class Language:
    def __init__(self, filepath):
        with open(filepath, "r") as f:
            self.words = [line.strip() for line in f if line.strip()]

    def generate_sentence(self, num_words, separator=" "):
        return separator.join(random.choices(self.words, k=num_words))

    def generate_corpus(self, num_words, filepath, separator=""):
        corpus = self.generate_sentence(num_words, separator=separator)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(corpus)


