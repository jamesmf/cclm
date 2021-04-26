from cclm import augmentation
from cclm.preprocessing import Preprocessor
from cclm.pretrainers.base_cl_pretrainer import BasePretrainer
from cclm.preprocessing import Preprocessor
from cclm.models import CCLMModelBase
from cclm.augmentation import Augmentor
from datasets import load_dataset
import tensorflow as tf

dataset = load_dataset("ag_news", cache_dir="/app/cclm/.datasets")["train"]["text"]

augmentor = Augmentor()

prep = Preprocessor()
prep.fit(dataset)
base = CCLMModelBase(preprocessor=prep)
bp = BasePretrainer(base=base, augmentor=augmentor)
bp.model.compile(
    tf.keras.optimizers.Adam(lr=0.001),
    tf.keras.losses.SparseCategoricalCrossentropy(),
)
gen = bp.generator(dataset)
x, y = next(gen)
bp.model.fit(gen, steps_per_epoch=100, epochs=300)