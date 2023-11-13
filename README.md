# yarn-mistral-flax
An implementation of yarn-mistral-7B in flax. This implementation is based on the [pytorch version uploaded to huggingface](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k
).


Nous-Yarn-Mistral-7b-128k is a state-of-the-art language model for long context, further pretrained on long context data for 1500 steps using the YaRN extension method. It is an extension of Mistral-7B-v0.1 and supports a 128k token context window.

From the abstract of the original [arxiv submission](https://arxiv.org/abs/2309.00071):


"Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset."

# Visualizing the improvements of yarn
