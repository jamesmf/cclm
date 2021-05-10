## CCLM

### Composable, Character-Level Models

#### Why `cclm`?


The goal of `cclm` is to make the deep learning model development process modular by providing abstractions for structuring a computational graph.

If we think of the ML lifecycle as producing a usable class `Model` that consumers can call on `input` to get `output`, then comparing the model training process to human-led software development highlights some big differences. For instance often when we retrain models, we usually change the whole model at once - imagine a developer telling you every commit they made touched every line of code in the package. Similarly, using a pretrained model is like using a 'batteries included' framework: you likely end up inheriting a good deal of functionality you don't require, and it may be hard to customize. These differences suggest that there may be changes that could make it easier to manage deep learning model development, particularly as models continue to explode in size.

#### How does it work?

The way `cclm` aims to achieve the above is by making the model building process composable. There are many ways to pretrain a model on text, and infinite corpora on which to train, and each application has different needs.

`cclm` makes it possible to define a `base` input on which to build many different computational graphs, then combine them. For instance, if there is a standard, published `cclm` model trained with masked language modeling (MLM) on (`wikitext` + `bookcorpus`), you might start with that, but add a second component to that model that uses the same `base`, but is pretrained to extract entities from `wiki-ner`. By combining the two pretrained components with a `ComposedModel`, you get a model with information from both tasks that you can then use as a starting point for your downstream task.

Common model components will be published onto the `cclm-shelf` to make it simple to mix and match capabilities.

The choice to emphasize character-level rather than arbitrary tokenization schemes is to make the input as generically useful across tasks as possible. Character-level input also makes it simpler to add realistic typos/noise to make models more robust to imperfect inputs.


#### Basic concepts

The main output of a training job with `cclm` is a `ComposedModel`, which consists of a `Preprocessor` that turns text into a vector[int], a base model that embeds that vector input, and one or more models that accept the output of the embedder. The `ComposedModel` concatenates the output from those models together to produce its final output.

The package uses `datasets` and `tokenizers` from `huggingface` for a standard interface and to benefit from their great framework- but to fit models and preprocessors, you can also pass a `List[str]` directly.

To start, you need a `Preprocessor`.

```python
from cclm.preprocessing import Preprocessor

prep = Preprocessor()  # set max_example_len to specify a maximum input length
prep.fit(dataset) # defines the model's vocabulary (character-level)
```

Once you have that, you can create a `CCLMModelBase`, which is the common base on which all the separate models will sit. This is a flexible class primarily responsible for holding a model that embeds a sequence of integers (representing characters) into a space the components expect. For more complicated setups, the `CCLMModelBase` could have a `ComposedModel` as a `.embedder`

```python
from cclm.models import CCLMModelBase

base = CCLMModelBase(preprocessor=prep)
```

The base doesn't need to be fit, as you can fit it while you do your first pretraining task.

Now you're ready to build your first model using a pretraining task (here masked language modeling)

```python
from cclm.pretraining import MaskedLanguagePretrainer

pretrainer = MaskedLanguagePretrainer(base=base)
pretrainer.fit(dataset, epochs=10)
```

The `MaskedLanguagePretrainer` defines a transformer model (which uses strided convolutions to reduce the size before the transformer layer, then upsamples to match the original size), and calling `.fit()` will use the `Preprocessor` associated with the `base` to produce masked inputs and try to identify the missing input token(s) using `sampled_softmax` loss or negative sampling. This is just one example of a pretraining task, but others can be found in `cclm.pretrainers`.

Once you've trained one or more models using `Pretrainer` objects, you can compose them together into one model.

```python
composed = ComposedModel(base, [pretrainer_a.model, pretrainer_b.model])
```

You can then use `composed.model(x)` to embed input

```python
x = prep.string_to_array("cclm is neat", prep.max_example_len)
emb = composed.model(x)   # has shape (1, prep.max_example_len, pretrainer_a_model_shape[-1]+pretrainer_b_model_shape[-1])
```

... or create a new model with something like

```python
# pool the output across the character dimension
gmp = tf.keras.layers.GlobalMaxPool1D()
# add a classification head on top
d = tf.keras.layers.Dense(1, activation="sigmoid")
keras_model = tf.keras.Model(composed.model.input, d(gmp(composed.model.output)))
```