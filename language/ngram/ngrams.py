import os
import torch
from collections import Counter

INPUT = "../fineweb/fineweb.txt"
OUTPUT_DIR = ""
N_VALUES = [1, 2, 3, 4]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for n in N_VALUES:
        counts = Counter()
        for line in lines:
            for i in range(len(line) - n + 1):
                counts[line[i:i+n]] += 1

        tensor = torch.zeros([26] * n, dtype=torch.float32)
        for ngram, count in counts.items():
            idx = tuple(ord(c) - 65 for c in ngram)
            tensor[idx] = count

        total = tensor.sum()
        probs = torch.clamp(tensor / total, min=0.01 / total)
        log_probs = torch.log10(probs)

        torch.save(log_probs, os.path.join(output_dir, f"{n}grams.pth"))
        print(f"Saved {n}grams.pth")

if __name__ == "__main__":
    main()
