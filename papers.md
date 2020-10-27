# Papers

Here I list all the papers that I read, together with summarization.

## Computer Vision

### [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
The three popular model scaling technique are width, height and depth scaling, each of them is preformed independently. In the paper, the authors scale the model by all three factors, but satisfy a constrain. The core layer in the EfficientNet model is mobile inverted bottleneck layer (MBConv), which was introduced in MobileNetV2. The model EfficientNetB7 has 66M parameters but achieved 84.4% top-1 accuracy on ImageNet, while the model GPipe with 557M parameters achieved 84.3%.

## Natural Language Processing

### [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
Attention mechanism had greatly improved the results on machine translation. This Luong's attention used $h_t$ to compute instead of $h_{t-1}$. It is also called dot-product attention, which is used in the Transformer model. The attention used matrix dot product, which performed better on today's GPU.

### [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
Recurrent neural network has two problems: it is vulnerable to vanishing/exploding gradients, and it takes a long time to train. Since the introduction of Attention mechanism, it had boosted the BLEU score a lot. The authors raised a question: if Attention helps the decoder to "see" all the states in the encoder, do we still need RNN? They introduced the Transformer model, which fully used Attention mechanism called Multi-head attention, together with feedforward neural network, positional encoding, they achieved SOTA results on machine translation.

### [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
The authors stated two problems with Vietnamese language modelling: Vietnamese uses space to seperate syllabus, and the Vietnamese Wikipedia corpus is only ~1GB. The authors trained the BERT model (based on RoBERTa), but used a ~20GB word-level corpus, and achieved SOTA on POS, Dependency parsing, NLI, NER.

### [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
The model is called GPT. The authors used the Transformer model. First it was unsupervised pre-training by using the Transformer decoder for the language model, then it was supervised fine-tuning on specific tasks. The model achieved SOTA results on 9 out of 12 dataset, which were NLI, Question Answering and Commonsense Reasoning, Semantic Similiarity and Text Classification tasks. The authors also noted that increasing the complexity of the layers indeed increased the accuracy.

## Others
These are the others paper that I read but I have not fully understood, or I am reading. I will take a visit later.
### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
### [Quantum supremacy using a programmable superconducting processor](https://www.nature.com/articles/s41586-019-1666-5)
### [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
### [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
### [Algorithms for hyper-parameter optimization](http://papers.nips.cc/paper/4443-algorithms-for-hyper)
### [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
### [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
### [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)
### [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)
### [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
### [NeuMiss networks: differential programming for supervised learning with missing values](https://arxiv.org/abs/2007.01627)
### [Untangling tradeoffs between recurrence and self-attention in neural networks](https://arxiv.org/abs/2006.09471)
### [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
