import regex as re
import json

class BPETokenizer:
    def __init__(self, pattern=None, special_tokens=None):
        self.vocab = {}
        self.merges = {}
        self.pattern = pattern or r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = special_tokens or {}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    @staticmethod
    def _most_common_pairs(tokens, counts=None):
        pair_counts = {} if counts is None else counts
        for i in range(1, len(tokens)):
            pair = (tokens[i-1], tokens[i])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts
    
    @staticmethod
    def _merge(ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text, vocab_size, verbose=False):
        num_special = len(self.special_tokens)
        num_merges = vocab_size - 256 - num_special
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        self.merges = {}
        self.vocab = {idx + num_special: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                self._most_common_pairs(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = num_special + 256 + i
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merging {pair} into a new token {idx}")

    def set_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text, allowed_special="all"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self._encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._most_common_pairs(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def save(self, vocab_path='./tokenizer/vocab.json', merges_path='./tokenizer/merges.txt'):
        with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
            vocab_dict = {idx: self.vocab[idx].decode('utf-8') for idx in self.vocab}
            json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=4)
        
        with open(merges_path, 'w', encoding='utf-8') as merges_file:
            for pair, idx in self.merges.items():
                merges_file.write(f"{pair[0]} {pair[1]} -> {idx}\n")

    def load(self, vocab_path='./tokenizer/vocab.json', merges_path='./tokenizer/merges.txt'):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab_dict = json.load(vocab_file)
            self.vocab = {int(idx): bytes(val, 'utf-8') for idx, val in vocab_dict.items()}
        
        self.merges = {}
        with open(merges_path, 'r', encoding='utf-8') as merges_file:
            for line in merges_file:
                parts = line.strip().split(' -> ')
                if len(parts) != 2:
                    continue
                pair, idx = parts[0], parts[1]
                idx = int(idx)
                token1, token2 = map(int, pair.split())
                self.merges[(token1, token2)] = idx
        
def main():
    special_tokens = {
        "<s>": 0,
        "</s>": 1,
        "<unk>": 2,
        "<pad>": 3,
        "<im_start>": 4,
        "<im_end>": 5,
        "<fim_prefix>": 6,
        "<fim_middle>": 7,
        "<fim_suffix>":8,
        "<mask>":9,
    }
    
    tokenizer = BPETokenizer()
    tokenizer.set_special_tokens(special_tokens)

    with open('./data/catalan_text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    tokenizer.train(text, vocab_size=65536, verbose=True)

    tokenizer.save('vocab.json', 'merges.txt')

if __name__ == "__main__":
    main()