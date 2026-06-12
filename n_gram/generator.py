import torch

def generate_ngram_counts(corpus_filepath, alphabet, n):
    with open(corpus_filepath, "r", encoding="utf-8") as f:
        corpus = f.read().upper()
    
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    corpus_indices = [char_to_idx[c] for c in corpus if c in char_to_idx]
    corpus_tensor = torch.tensor(corpus_indices, dtype=torch.long)
    
    unfolded = corpus_tensor.unfold(0, n, 1)
    K = len(alphabet)
    powers = torch.tensor([K ** (n - 1 - i) for i in range(n)], dtype=torch.long)
    flat_indices = (unfolded * powers).sum(dim=1)
    
    return torch.bincount(flat_indices, minlength=K ** n).view(*[K] * n)

def save_ngram_counts(tensor, filepath):
    torch.save(tensor, filepath)

def load_ngram_counts(filepath):
    return torch.load(filepath)

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base_dir, "language", "corpus3.txt")
    counts = generate_ngram_counts(corpus_path, "ABC", 3)
    print("Shape:", counts.shape)
    print(counts)
    save_ngram_counts(counts, os.path.join(base_dir, "n_gram", "counts3.pth"))

