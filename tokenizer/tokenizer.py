from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()


special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<im_start>", "<im_end>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<mask>"]


files = ["./data/catalan_text.txt"]

tokenizer.train(files=files, vocab_size=500, show_progress = True, special_tokens=special_tokens)


tokenizer.save("./tokenizer/mini.tokenizer.json")
tokenizer.model.save("tokenizer")
