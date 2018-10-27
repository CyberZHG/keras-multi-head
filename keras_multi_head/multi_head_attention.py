import keras
import keras.backend as K
from keras_self_attention import ScaledDotProductAttention


class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 kernel_initializer='glorot_normal',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 kernel_activation='relu',
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param kernel_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param kernel_activation: Activations for linear mappings.
        :param feature_dim: The dimension of input feature.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.kernel_activation = keras.activations.get(kernel_activation)

        self.kernels = {name: None for name in ['Wq', 'Wk', 'Wv', 'Wo']}
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'kernel_activation': self.kernel_activation,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        for name in ['Wq', 'Wk', 'Wv', 'Wo']:
            self.kernels[name] = self.add_weight(
                shape=(feature_dim, feature_dim),
                initializer=self.kernel_initializer,
                name='%s_%s' % (self.name, name),
            )
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        feature_dim = K.shape(inputs)[-1]
        head_dim = feature_dim // self.head_num
        q = self.kernel_activation(K.dot(inputs, self.kernels['Wq']))
        k = self.kernel_activation(K.dot(inputs, self.kernels['Wk']))
        v = self.kernel_activation(K.dot(inputs, self.kernels['Wv']))
        outputs = []
        for i in range(self.head_num):
            begin, end = i * head_dim, (i + 1) * head_dim
            outputs.append(ScaledDotProductAttention(name='%s-Att-%d' % (self.name, i + 1))([
                q[:, :, begin:end],
                k[:, :, begin:end],
                v[:, :, begin:end],
            ]))
        return self.kernel_activation(K.dot(K.concatenate(outputs), self.kernels['Wo']))
