import copy
import keras
import keras.backend as K


class MultiHead(keras.layers.Wrapper):

    def __init__(self,
                 layer,
                 layer_num,
                 **kwargs):
        """Initialize the wrapper layer.

        :param layer: The layer to be duplicated.
        :param layer_num: The number of duplicated layers.
        :param kwargs: Arguments for parent.
        """
        self.layer = layer
        self.layers = []
        self.layer_num = layer_num
        super(MultiHead, self).__init__(layer, **kwargs)

    def get_config(self):
        config = {'layer_num': self.layer_num}
        base_config = super(MultiHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if type(input_shape) == list:
            self.input_spec = list(map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        else:
            self.input_spec = keras.engine.InputSpec(shape=input_shape)
        if not self.layers:
            self.layers = [copy.copy(self.layer) for _ in range(self.layer_num)]
            for i, layer in enumerate(self.layers):
                layer.name = layer.name + '_%d' % (i + 1)
                layer.build(input_shape)
        super(MultiHead, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        child_output_shape = self.layers[0].compute_output_shape(input_shape)
        return child_output_shape + (self.layer_num,)

    def compute_mask(self, inputs, mask=None):
        return self.layers[0].compute_mask(inputs, mask)

    def call(self, inputs, mask=None):
        outputs = [K.expand_dims(layer.call(inputs, mask)) for layer in self.layers]
        return K.concatenate(outputs, axis=-1)
