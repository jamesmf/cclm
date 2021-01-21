## CCLM

### Composable, Character-Level Models

#### What are the goals of the project?

1) Modularity: Fine-tuning large language models is expensive. `cclm` seeks to decompose models into subcomponents that can be readily mixed and matched, allowing for a wider variety of sizes, architectures, and pretraining methods. Rather than fine-tuning a huge model on your own data, fit a smaller one on your dataset and combine it with off-the-shelf models.

2) Character-level input: Many corpora used in pretraining are clean and typo-free, but a lot of user-focused inputs aren't - leaving you at a disadvantage if your tokenization scheme isn't flexible enough. Using characters as input also makes it simple define many 'heads' of a model with the same input space.

3) Ease of use: It should be quick to get started and easy to deploy. 


#### How does it work?

The way `cclm` hopes to achieve the above is by making the model building process composable. There are many ways to pretrain a model on text, and infinite corpora on which to train, and each application has different needs.

`cclm` makes it possible to define a 'base' input on which to build many different computational graphs, then combine them. For instance, if there is a standard, published `cclm` model trained with masked language modeling (MLM) on (`wikitext` + `bookcorpus`), you might start with that, but add a second 'tower' to that model that uses the same 'base', but is pretrained to extract entities from `wiki-ner`. By combining the two pretrained 'towers', you get a model with information from both tasks that you can then use as a starting point for your downstream model.

As the package matures, the goal is to make available many pretraining methods (starting with Masked Language Modeling) and to publish standard pretrained models (like huggingface/transformers, spacy, tensorflowhub, ...).


#### Basic concepts

The main output of a training job with `cclm` is a `ComposedModel`, which consists of a preprocessor that turns text into a vector[int], a base model that embeds that vector input, and one or more models that accept the output of the embedder. The `ComposedModel` concatenates the output from those models together to produce its final output.

The package uses `datasets` and `tokenizers` from `huggingface` for a standard interface - but to fit models, you can pass a `List[str]` directly.

To start, you need a `Preprocessor`. Currently, there is only an `MLMPreprocessor` that computes extra data at training time for its pretraining task, but that is subject to change.

```
from cclm.preprocessing import MLMPreprocessor

prep = MLMPreprocessor()  # set max_example_len to specify a maximum input length
prep.fit(dataset) # defines the characters the model knows about and the output tokens for MLM
```

Once you have that, you can create a `CCLMModelBase`, which is the common base on which all the separate models will sit.

```
from cclm.models import CCLMModelBase

base = CCLMModelBase(preprocessor=prep)
```

The base doesn't need to be fit, as you can fit it while you do your first pretraining task.

Now you're ready to build your first model using a pretraining task (here masked language modeling)

```
from cclm.pretraining import MaskedLanguagePretrainer

pretrainer = MaskedLanguagePretrainer(
    base=base,
    downsample_factor=16,  # how much we want to reduce the length of the input sequence
    n_strided_convs=4,  # how many strided conv layers we have. stride_len**n_strided_convs must == downsample_factor
)

pretrainer.fit(dataset, epochs=10)
```

The `MaskedLanguagePretrainer` defines a transformer model (which uses strided convolutions to reduce the size before the transformer layer, then upsamples to match the original size), and calling `.fit()` will use the `MLMPreprocessor` associated with the `base` to produce masked inputs and try to identify the missing input token(s) using `sampled_softmax` loss.

Once you've trained one or more models with `Pretrainer` objects, you can compose them together into one model.

```
composed = ComposedModel(base, [pretrainer_a.model, pretrainer_b.model])
```

You can then use `composed.model(x)` to embed input

```
x = prep.string_to_array("cclm SURE is useful!!", prep.max_example_len)
emb = composed.model(x)   # has shape (1, prep.max_example_len, pretrainer_a_model_shape[-1]+pretrainer_b_model_shape[-1])
```

... or create a new model with something like

```
# pool the output across the character dimension
gmp = tf.keras.layers.GlobalMaxPool1D()
# add a classification head on top
d = tf.keras.layers.Dense(1, activation="sigmoid")
keras_model = tf.keras.Model(composed.model.input, d(gmp(composed.model.output)))
```