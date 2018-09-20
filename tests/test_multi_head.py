import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_self_attention import SeqSelfAttention as Attention
from keras_piecewise_pooling import PiecewisePooling1D
from keras_multi_head import MultiHead


class TestMultiHead(unittest.TestCase):

    def test_multi_attention(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=5, output_dim=3, mask_zero=True, name='Embed'))
        model.add(MultiHead(Attention(name='Attention'), layer_num=5, name='Multi-Head-Attention'))
        model.build()
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mae, metrics=[keras.metrics.mae])
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'SeqSelfAttention': Attention,
            'MultiHead': MultiHead,
        })
        model.summary()
        predicts = model.predict(np.random.randint(0, 4, (5, 7)))
        self.assertEqual((5, 7, 3, 5), predicts.shape)

    def test_fake_multi_pooling(self):
        data = [
            [1, 3, 2, 4],
            [2, 8, 3, 5],
        ]
        positions = [
            [1, 3],
            [2, 4],
        ]
        data_input = keras.layers.Input(shape=(4,), name='Input-Data')
        pos_input = keras.layers.Input(shape=(2,), name='Input-Pos')
        pooling = MultiHead(PiecewisePooling1D(), layer_num=2, name='Multi-Head-Pooling')([data_input, pos_input])
        model = keras.models.Model(inputs=[data_input, pos_input], outputs=pooling)
        model.summary()
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [[1.0, 1.0], [3.0, 3.0]],
            [[8.0, 8.0], [5.0, 5.0]],
        ]
        self.assertEqual(expected, predicts)
