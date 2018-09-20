import os
import tempfile
import random
import unittest
import keras
import numpy as np
from keras_self_attention import SeqSelfAttention as Attention
from keras_piecewise_pooling import PiecewisePooling1D
from keras_multi_head import MultiHead


class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, _, mask=None):
        return mask


class TestMultiHead(unittest.TestCase):

    @staticmethod
    def data_generator(batch_size=32):
        while True:
            max_len = random.randint(5, 10)
            data = np.zeros((batch_size, max_len))
            tag = np.zeros(batch_size, dtype='int32')
            for i in range(batch_size):
                datum_len = random.randint(1, max_len - 1)
                total = 0
                for j in range(datum_len):
                    data[i, j] = random.randint(1, 4)
                    total += data[i, j]
                tag[i] = total % 2
            yield data, tag

    def test_multi_attention(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=5, output_dim=3, mask_zero=True, name='Embed'))
        model.add(MultiHead(Attention(name='Attention'), layer_num=5, name='Multi-Head-Attention'))
        model.add(keras.layers.TimeDistributed(MaskFlatten(), name='Flatten'))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=8), name='Bi-GRU'))
        model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
        model.build()
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )

        model.fit_generator(
            generator=self.data_generator(),
            steps_per_epoch=100,
            epochs=100,
            validation_data=self.data_generator(),
            validation_steps=10,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
            ],
        )
        model.layers[1].set_weights(model.layers[1].get_weights())

        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'MaskFlatten': MaskFlatten,
            'SeqSelfAttention': Attention,
            'MultiHead': MultiHead,
        })
        model.summary()
        for data, tag in self.data_generator():
            predicts = model.predict(data)
            predicts = np.argmax(predicts, axis=-1)
            self.assertGreaterEqual(np.sum(tag == predicts), 30)
            break

    def test_multi_pooling(self):
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
        pooling = MultiHead(
            [
                PiecewisePooling1D(pool_type=PiecewisePooling1D.POOL_TYPE_MAX),
                PiecewisePooling1D(pool_type=PiecewisePooling1D.POOL_TYPE_AVERAGE),
            ],
            name='Multi-Head-Pooling',
        )([data_input, pos_input])
        model = keras.models.Model(inputs=[data_input, pos_input], outputs=pooling)
        model.summary()
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [[1.0, 1.0], [3.0, 2.5]],
            [[8.0, 5.0], [5.0, 4.0]],
        ]
        self.assertTrue(np.allclose(expected, predicts))

        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        custom_objects = PiecewisePooling1D.get_custom_objects()
        custom_objects['MultiHead'] = MultiHead
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        predicts = model.predict([np.asarray(data), np.asarray(positions)]).tolist()
        expected = [
            [[1.0, 1.0], [3.0, 2.5]],
            [[8.0, 5.0], [5.0, 4.0]],
        ]
        self.assertTrue(np.allclose(expected, predicts))

    def test_multi_cnn(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=5, output_dim=3, name='Embed'))
        model.add(MultiHead([
            keras.layers.Conv1D(filters=32, kernel_size=3, padding='same'),
            keras.layers.Conv1D(filters=32, kernel_size=5, padding='same'),
            keras.layers.Conv1D(filters=32, kernel_size=7, padding='same'),
        ], name='Multi-Head-Attention'))
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten(), name='Flatten'))
        model.add(keras.layers.Bidirectional(keras.layers.GRU(units=8), name='Bi-GRU'))
        model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
        model.build()
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=[keras.metrics.sparse_categorical_accuracy],
        )

        model.fit_generator(
            generator=self.data_generator(),
            steps_per_epoch=100,
            epochs=100,
            validation_data=self.data_generator(),
            validation_steps=10,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=5),
            ],
        )
        model.layers[1].set_weights(model.layers[1].get_weights())

        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'MultiHead': MultiHead})
        model.summary()
        for data, tag in self.data_generator():
            predicts = model.predict(data)
            predicts = np.argmax(predicts, axis=-1)
            self.assertGreaterEqual(np.sum(tag == predicts), 30)
            break
