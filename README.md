<!--- BADGES: START --->
[![HF Models](<https://img.shields.io/badge/%F0%9F%A4%97-base model-yellow>)](https://huggingface.co/pauhidalgoo/cucafera)
[![HF Models](<https://img.shields.io/badge/%F0%9F%A4%97-instruct model-yellow>)](https://huggingface.co/pauhidalgoo/cucafera-instruct)
[![HF Models](<https://img.shields.io/badge/%F0%9F%A4%97-chat model-yellow>)](https://huggingface.co/pauhidalgoo/cucafera-chat)


# cucafera 🔥🐲
> a Catalan LLM

<div align="center">
    <img src="./media/modelimage.png" alt="Description of Image">
</div>

## Description

**Cucafera** is a Large Language Model, inspired by the LLAMA architecture (specially LLAMA3). It is very small (in comparison with other models), with just 244M parameters, but it *sometimes* gets the job done.

This model is the result of a personal project which involved creating high quality datasets, coding the models architecture in Pytorch and training/finetunning it. My collegue @rogerbaiges did the same thing with his [CatGPT](https://github.com/rogerbaiges/CatGPT).

The goal of the project was to experiment with LLMs and for educational purposes. It its pretty simple, but it is able to generate good text in catalan. You can try it on [HuggingFace](https://huggingface.co/pauhidalgoo/cucafera)

## Creation process

One of our goals was to be totally transparent with all the steps we have taken to create this model. We encourage others to try to do the same things that we did and to share their results!

For starters, the first step was determining the datasets for training and pretraining. In my case, they consisted on the [patufet](https://github.com/pauhidalgoo/patufet/tree/main) collection of datsets, also created by us.

### Training

The dataset used was [patufet-educat](https://huggingface.co/datasets/pauhidalgoo/patufet-educat), which is CulturaX (mc4 + OSCAR) filtered by educational content.

In total, I did 11007 steps of 0.5M tokens, which end up being 5.5B of tokens. The [Chinchilla paper](https://arxiv.org/abs/2203.15556) stated that the optimal model scaling was around training 20 times the number of parameters. In our case, it was more around x25 (more in line with the [chinchilla replication attempt by Epoch AI](https://www.arxiv.org/abs/2404.10102) and [DeepSeek scaling laws](https://arxiv.org/abs/2401.02954), considering that we had "high quality"). 

The time it took to train was 1.5 days, with a cost of 28.8$. Training was done on a single A100 80GB GPU on VastAI. 

If we had access to more money and resources, we would have trained it more, since it still showed signs of improvement and we feel like it is a bit undertrained. Also, the learning-rate can be set higher (we didn't do it because we had some problems at first with a bad model version and exploding gradients :/ )

### Finetunning

Finetunning was done using Huggingface's [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer). It was really quick and easy to do, so anyone can do it. The dataset consisted of a mix of different instructions datasets from patufet (feel free to check the code or HF) and the conversa dataset.

We opted for the ChatML format, which is pretty standard, and uses the <im_start> and <im_end> tags.

This process took around 1 hour on a A100 80GB GPU (0.80$).

## Model architecture

- 244 M parameters
- 65536 vocab size
- 30 layers
- 2048 context length
- 768 embedding size
- 4 key value heads, 8 query heads (GQA)

It follows more or less the LLAMA 3 architecture (with RoPE, GQA...) and uses GeGLU activation function.

The tokenizer used (also in this repo) is a BPE tokenizer.

## Limitations

Since it is a small model, it has significant limitations.
- Incorrect facts or information
- The instruct/chat version sometimes doesn't follow instructions well.
- It only knows Catalan (it was fully trained in this language)
- RLHF/DPO is still missing, so there can be some safety issues
- You CAN'T use this model to compete with Gemini API or Google AI services, since it was trained on Gemini-1.5-flash generated data.

## Evaluation
Evaluation was done using Eleuther AI lm evaluation harness on the Catalan tasks.

|        Tasks        |Version|     Filter      |n-shot|  Metric   |   | Value  |   |Stderr|
|---------------------|-------|-----------------|-----:|-----------|---|-------:|---|------|
|catalan_bench        |    N/A|                 |      |           |   |        |   |      |
| - arc_ca_challenge  |      1|none             |     0|acc        |↑  |  0.2295|±  |0.0123|
|                     |       |none             |     0|acc_norm   |↑  |  0.2534|±  |0.0127|
| - arc_ca_easy       |      1|none             |     0|acc        |↑  |  0.4238|±  |0.0101|
|                     |       |none             |     0|acc_norm   |↑  |  0.4108|±  |0.0101|
| - belebele_cat_Latn |      0|none             |     0|acc        |↑  |  0.2289|±  |0.0140|
|                     |       |none             |     0|acc_norm   |↑  |  0.2289|±  |0.0140|
| - cabreu_abstractive|      1|none             |     0|bleu       |↑  |  2.8684|±  |0.2838|
|                     |       |none             |     0|rouge1     |↑  |  0.1769|±  |   N/A|
| - cabreu_extractive |      1|none             |     0|bleu       |↑  |  1.3677|±  |0.1970|
|                     |       |none             |     0|rouge1     |↑  |  0.1829|±  |   N/A|
| - cabreu_extreme    |      1|none             |     0|bleu       |↑  |  1.3403|±  |0.1363|
|                     |       |none             |     0|rouge1     |↑  |  0.1217|±  |   N/A|
| - catalanqa         |      1|none             |     0|exact_match|↑  |  0.0037|±  |0.0013|
|                     |       |none             |     0|f1         |↑  |  0.0991|±  |0.0032|
| - catcola           |      1|none             |     0|acc        |↑  |  0.2967|±  |0.0143|
|                     |       |none             |     0|mcc        |↑  |  0.0723|±  |0.0175|
| - copa_ca           |      1|none             |     0|acc        |↑  |  0.6140|±  |0.0218|
| - coqcat            |      1|none             |     0|em         |↑  |  0.1433|±  |0.0143|
|                     |       |none             |     0|f1         |↑  |  0.2227|±  |0.0142|
| - flores_ca         |      1|none             |      |bleu       |↑  |  0.5934|±  |0.0263|
|  - flores_ca-de     |      1|none             |     0|bleu       |↑  |  0.4539|±  |0.1376|
|                     |       |none             |     0|chrf       |↑  | 14.5978|±  |0.1763|
|                     |       |none             |     0|ter        |↓  |123.3869|±  |3.3182|
|  - flores_ca-en     |      1|none             |     0|bleu       |↑  |  0.4964|±  |0.0869|
|                     |       |none             |     0|chrf       |↑  | 17.4055|±  |0.2697|
|                     |       |none             |     0|ter        |↓  |128.1220|±  |3.5590|
|  - flores_ca-es     |      1|none             |     0|bleu       |↑  |  1.7610|±  |0.1361|
|                     |       |none             |     0|chrf       |↑  | 24.6096|±  |0.3309|
|                     |       |none             |     0|ter        |↓  |105.9315|±  |2.6560|
|  - flores_ca-eu     |      1|none             |     0|bleu       |↑  |  0.2846|±  |0.0797|
|                     |       |none             |     0|chrf       |↑  | 15.3801|±  |0.1941|
|                     |       |none             |     0|ter        |↓  |139.4097|±  |2.8295|
|  - flores_ca-fr     |      1|none             |     0|bleu       |↑  |  1.0479|±  |0.1424|
|                     |       |none             |     0|chrf       |↑  | 20.7805|±  |0.2760|
|                     |       |none             |     0|ter        |↓  |110.5043|±  |2.3637|
|  - flores_ca-gl     |      1|none             |     0|bleu       |↑  |  1.3179|±  |0.1623|
|                     |       |none             |     0|chrf       |↑  | 23.8786|±  |0.3082|
|                     |       |none             |     0|ter        |↓  |109.5959|±  |2.3746|
|  - flores_ca-it     |      1|none             |     0|bleu       |↑  |  0.8669|±  |0.1187|
|                     |       |none             |     0|chrf       |↑  | 20.8593|±  |0.2580|
|                     |       |none             |     0|ter        |↓  |115.5321|±  |2.9987|
|  - flores_ca-pt     |      1|none             |     0|bleu       |↑  |  1.0632|±  |0.1258|
|                     |       |none             |     0|chrf       |↑  | 21.7723|±  |0.2964|
|                     |       |none             |     0|ter        |↓  |112.2880|±  |2.3862|
|  - flores_de-ca     |      1|none             |     0|bleu       |↑  |  0.0027|±  |0.0023|
|                     |       |none             |     0|chrf       |↑  |  1.5722|±  |0.1490|
|                     |       |none             |     0|ter        |↓  |107.8674|±  |1.5172|
|  - flores_en-ca     |      1|none             |     0|bleu       |↑  |  0.1991|±  |0.0702|
|                     |       |none             |     0|chrf       |↑  |  5.9145|±  |0.2289|
|                     |       |none             |     0|ter        |↓  |108.2112|±  |2.2993|
|  - flores_es-ca     |      1|none             |     0|bleu       |↑  |  1.1257|±  |0.1499|
|                     |       |none             |     0|chrf       |↑  | 12.9228|±  |0.3279|
|                     |       |none             |     0|ter        |↓  |141.4941|±  |5.1332|
|  - flores_eu-ca     |      1|none             |     0|bleu       |↑  |  0.0192|±  |0.0093|
|                     |       |none             |     0|chrf       |↑  |  2.3988|±  |0.1565|
|                     |       |none             |     0|ter        |↓  |114.8342|±  |3.1420|
|  - flores_fr-ca     |      1|none             |     0|bleu       |↑  |  0.1376|±  |0.0481|
|                     |       |none             |     0|chrf       |↑  |  3.8497|±  |0.2042|
|                     |       |none             |     0|ter        |↓  |116.0336|±  |2.7499|
|  - flores_gl-ca     |      1|none             |     0|bleu       |↑  |  0.3380|±  |0.1102|
|                     |       |none             |     0|chrf       |↑  |  6.9272|±  |0.2542|
|                     |       |none             |     0|ter        |↓  |128.3913|±  |3.6808|
|  - flores_it-ca     |      1|none             |     0|bleu       |↑  |  0.2636|±  |0.0711|
|                     |       |none             |     0|chrf       |↑  |  6.2937|±  |0.2377|
|                     |       |none             |     0|ter        |↓  |141.7929|±  |4.9810|
|  - flores_pt-ca     |      1|none             |     0|bleu       |↑  |  0.1170|±  |0.0472|
|                     |       |none             |     0|chrf       |↑  |  3.9389|±  |0.2030|
|                     |       |none             |     0|ter        |↓  |120.9988|±  |3.4963|
| - mgsm_direct_ca    |      1|remove_whitespace|     0|exact_match|↑  |  0.0000|±  |     0|
| - openbookqa_ca     |      1|none             |     0|acc        |↑  |  0.2120|±  |0.0183|
|                     |       |none             |     0|acc_norm   |↑  |  0.3160|±  |0.0208|
| - parafraseja       |      1|none             |     0|acc        |↑  |  0.0000|±  |     0|
| - paws_ca           |      1|none             |     0|acc        |↑  |  0.5080|±  |0.0112|
| - phrases_ca-va     |      1|none             |     5|bleu       |↑  | 40.3898|±  |1.6151|
|                     |       |none             |     5|chrf       |↑  | 61.3771|±  |0.7088|
|                     |       |none             |     5|ter        |↓  | 53.8224|±  |2.6590|
| - phrases_va-ca     |      1|none             |     5|bleu       |↑  | 39.1577|±  |1.1630|
|                     |       |none             |     5|chrf       |↑  | 54.1087|±  |0.6933|
|                     |       |none             |     5|ter        |↓  | 60.5922|±  |1.9343|
| - piqa_ca           |      1|none             |     0|acc        |↑  |  0.5996|±  |0.0114|
|                     |       |none             |     0|acc_norm   |↑  |  0.6121|±  |0.0114|
| - siqa_ca           |      1|none             |     0|acc        |↑  |  0.3705|±  |0.0109|
| - teca              |      1|none             |     0|acc        |↑  |  0.3751|±  |0.0105|
| - wnli_ca           |      1|none             |     0|acc        |↑  |  0.5493|±  |0.0595|
| - xnli_ca           |      1|none             |     0|acc        |↑  |  0.4261|±  |0.0099|
| - xquad_ca          |      1|none             |     0|exact_match|↑  |  0.0067|±  |0.0024|
|                     |       |none             |     0|f1         |↑  |  0.0842|±  |0.0041|
| - xstorycloze_ca    |      1|none             |     0|acc        |↑  |  0.5817|±  |0.0127|

|   Groups   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------|------:|------|------|------|---|-----:|---|-----:|
| - flores_ca|      1|none  |      |bleu  |↑  |0.5934|±  |0.0263|

## Use the model

We recommend to use the HuggingFace library to download/test the model. You can find an example [here](./src/generate.py).

Running the [telegram_bot](./telegram/telegram_bot.py) file will set up a Telegram bot that uses the model. Remember to replace the TOKEN with your own.

## To-do
 - [ ] RLHF / DPO
 - [ ] Training a larger model
 - [ ] Training this model for longer
 - [ ] Quantize / optimize for CPU inference
