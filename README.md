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

The time it took to train was 1.5 days, with a cost of 28.8$. Training was done on a single GPU on VastAI. 

If we had access to more money and resources, we would have trained it more, since it still showed signs of improvement and we feel like it is a bit undertrained. Also, the learning-rate can be set higher (we didn't do it because we had some problems at first with a bad model version and exploding gradients :/ )

### Finetunning

Finetunning was done using Huggingface's [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer). It was really quick and easy to do, so anyone can do it. The dataset consisted of a mix of different instructions datasets from patufet (feel free to check the code or HF) and the conversa dataset..

We opted for the ChatML format, which is pretty standard, and uses the <im_start> and <im_end> tags.



------

*ReadMe in progress*  