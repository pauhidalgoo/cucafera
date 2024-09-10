from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("pauhidalgoo/cucafera-chat")
model = AutoModelForCausalLM.from_pretrained("pauhidalgoo/cucafera-chat")

genconf = GenerationConfig(
    max_length = 150,
    repetition_penalty = 1.2,
    temperature = 0.6,
    top_k = 50,
    top_p = 0.9,
    do_sample = True,

)
tokens = np.array(tokenizer.encode("""<|im_start|>user
Hola!<|im_end|>
<|im_start|>assistant
Bon dia!<|im_end|>
<|im_start|>user
Què és un pingüí?<|im_end|>
<|im_start|>assistant"""))
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0)
a = model.generate(tokens, genconf)
for response in range(len(a)):
    print("> ", tokenizer.decode(a[response].tolist()))