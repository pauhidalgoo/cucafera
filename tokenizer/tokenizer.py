from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)



tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)


special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<im_start>", "<im_end>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<mask>"]
trainer = BpeTrainer(vocab_size=65536, show_progress = True, special_tokens=special_tokens)
 
tokenizer.pre_tokenizer = Whitespace()

files = ["./data/catalan_text.txt"]

tokenizer.train(files, trainer)


tokenizer.save("./tokenizer/byte-level-bpe.tokenizer.json")
tokenizer.model.save("tokenizer")