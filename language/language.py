import random

class Language:
    def __init__(self, filepath):
        with open(filepath, "r") as f:
            self.words = [line.strip() for line in f if line.strip()]

    def generate_sentence(self, num_words, separator=" "):
        return separator.join(random.choices(self.words, k=num_words))


