import tiktoken
import torch
from model import GPTConfig, GPT
from utils import load_checkpoint
from torch.nn import functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

enc = tiktoken.get_encoding("gpt2")

num_return_sequences = 10
max_lenght = 1024

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#x = tokens.to('cuda')

model_config = GPTConfig(vocab_size=50304)

model = GPT(model_config)
model.to(device)

load_checkpoint()
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_lenght:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_lenght].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)