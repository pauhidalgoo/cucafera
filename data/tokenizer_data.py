from datasets import load_dataset
import os
import os

dataset = load_dataset("uonlp/CulturaX", "ca", split="train", use_auth_token=True, streaming=True)

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "catalan_text.txt")

text_accumulator = ""
max_size_mb = 50
max_size_bytes = max_size_mb * 1024 * 1024

for example in dataset:
    text_accumulator += example["text"] + "\n"
    if len(text_accumulator.encode('utf-8')) > max_size_bytes:
        break

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text_accumulator)

print(f"Saved {len(text_accumulator.encode('utf-8')) / (1024 * 1024):.2f} MB of text to {output_file}")
