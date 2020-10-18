### CCLM

#### Composable, Character-Level Models

##### What are the goals of the project?

1) Modularity: Fine-tuning BERT (or GPT, if you dare) is expensive. It should be easy to do custom pre-training, and no one should have to start with a billion parameters to get the benefit of pretrained models. With `cclm` you piece together what you need, and leave the other 900M parameters in the cloud.

2) Character-level input: Many corpora used in pretraining are clean and typo-free, but a lot of user-focused inputs aren't - leaving you at a disadvantage if your tokenization scheme isn't flexible enough. Using characters as input also makes it simple define many 'heads' of a model with the same input space.

3) Ease of use: It should be quick to get started and easy to deploy. No one wants to waste developer time and it hurts the environment to use a bigger model if a simpler one will do.


##### How does it work?

The way `cclm` hopes to achieve the above is by making the model building process composable. There are many ways to pretrain a model on text, and infinite corpora on which to train, and each application has different needs.

`cclm` makes it possible to define a 'base' input on which to build many different computational graphs, then combine them. For instance, if there is a standard, published `cclm` model trained with MLM on (`wikitext` + `bookcorpus`), you might start with that, but add a second 'head' to that model that uses the same 'base', but is pretrained to extract entities from `wiki-ner`. By combining the two pretrained 'towers', you get a model with information from both tasks that you can then use as a starting point for your downstream model.

