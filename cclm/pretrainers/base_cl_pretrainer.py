from .core import Pretrainer


class BasePretrainer(Pretrainer):
    """
    Pretrain a CCLMBase embedder with a masked-character-like model.

    The base of the model has a SpatialDropout1D after the embedder
    to encourage learning a spelling-invariant representation of text.

    This task is to take an input, apply noisy transformations
    (including the SpatialDropout1D), and predict the original input
    """

    def __init__(self):
        super().__init__()

    def get_model(self):
        """
        SDO1D + LSTM
        """
        return

    def fit(self):
        """
        self.model.fit(gen)
        """
        return

    def get_generator(self, data):
        """
        Return a generator over the dataset that vectorizes the text and applies
        the noisy transformation to the input.
        """
        return