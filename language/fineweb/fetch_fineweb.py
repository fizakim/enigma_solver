from datasets import load_dataset
import re
import os

ROWS = 10_000
ROW_LENGTH = 100
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineweb.txt")

dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, row in enumerate(dataset):
        if i >= ROWS:
            break
        clean = re.sub(r"[^A-Za-z]", "", row["text"]).upper()
        for j in range(0, len(clean), ROW_LENGTH):
            chunk = clean[j:j + ROW_LENGTH]
            if len(chunk) == ROW_LENGTH:
                f.write(chunk + "\n")

print(f"Done")
