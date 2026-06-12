import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from language.language import Language

def generate_corpus(word_file_path: str, n_lines: int, n_words: int) -> None:
    lang = Language(word_file_path)
    lines = [lang.generate_sentence(n_words, separator="") for i in range(n_lines)]
    corpus = "\n".join(lines)
    
    dir_name, filename = os.path.split(word_file_path)
    corpus_filename = filename.replace("words", "corpus")
    corpus_path = os.path.join(dir_name, corpus_filename)
    
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus)

if __name__ == "__main__":
    words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "words3.txt")
    generate_corpus(words_path, 100_000, 100)
