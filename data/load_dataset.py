import os
import multiprocess as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Carpeta a la que anirà
    local_dir = "./data/dasaset"
    # La versió més petita, la de 10B de tokens
    remote_name = "sample-10BT"
    # Mida de les parts (shard) en les que dividirem el dataset
    shard_size = int(1e8)

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Carreguem el dataset de huggingface
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # Tokenitzem utilitzant tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    def tokenize(doc):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    # Funció per escriure el fitxer (np.save no anava bé en windows)
    def write_datafile(filename, tokens_np):
        with open(filename, "wb") as f:
            f.write(tokens_np.tobytes())

    nprocs = max(1, os.cpu_count()//2)
    # Paral·lelitzem per agilitzar el procés
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    # tqdm és una llibreria per fer barres de progrés
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

if __name__ == '__main__':
    main()