# Keras Multi-Head

[![Version](https://img.shields.io/pypi/v/keras-multi-head.svg)](https://pypi.org/project/keras-multi-head/)
![License](https://img.shields.io/pypi/l/keras-multi-head.svg)

将多个层横向放置在一起的封装：

![](https://user-images.githubusercontent.com/853842/45797517-867b8580-bcd8-11e8-9ec6-39d6508cf438.png)

## 安装

```bash
pip install keras-multi-head
```

## 使用

### 重复单一层

当输入的第一个参数只包含一个层时，`layer_num`决定了会复制多少相同配置的层并列到一起：

```python
from tensorflow import keras
from keras_multi_head import MultiHead


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=100, output_dim=20, name='Embedding'))
model.add(MultiHead(keras.layers.LSTM(units=32), layer_num=5, name='Multi-LSTMs'))
model.add(keras.layers.Flatten(name='Flatten'))
model.add(keras.layers.Dense(units=4, activation='softmax', name='Dense'))
model.build()
model.summary()
```

### 使用多种层

第一个参数也可以输出不同的层，但最终的输出必须要相同：

```python
from tensorflow import keras
from keras_multi_head import MultiHead


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=100, output_dim=20, name='Embedding'))
model.add(MultiHead([
    keras.layers.Conv1D(filters=32, kernel_size=3, padding='same'),
    keras.layers.Conv1D(filters=32, kernel_size=5, padding='same'),
    keras.layers.Conv1D(filters=32, kernel_size=7, padding='same'),
], name='Multi-CNNs'))
model.build()
model.summary()
```

### 输入线性变换

如果提供了`hidden_dim`参数，那么输出会被线性映射到不同的值。

### 正则化

![](https://user-images.githubusercontent.com/853842/45857922-8b4e4100-bd8d-11e8-905a-4eb07da31418.png)

正则化应用于层的可训练权重，目的是为了让平行的层提取出不同的特征，可以指定只使用其中一端权重进行正则化。如双向的LSTM包含6个权重，前三个属于前向传播，后三个属于后向传播。每组里第二个权重recurrent状态的计算，`units x 2`到`units x 3`部分负责计算cell states。如下是将前向传播的recurrent权重，后向传播recurrent权重的cell states部分进行正则化：

```python
from tensorflow import keras
from keras_multi_head import MultiHead


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=5, output_dim=3, name='Embed'))
model.add(MultiHead(
    layer=keras.layers.Bidirectional(keras.layers.LSTM(units=16), name='LSTM'),
    layer_num=5,
    reg_index=[1, 4],
    reg_slice=(slice(None, None), slice(32, 48)),
    reg_factor=0.1,
    name='Multi-Head-Attention',
))
model.add(keras.layers.Flatten(name='Flatten'))
model.add(keras.layers.Dense(units=2, activation='softmax', name='Dense'))
model.build()
```

* `reg_index`: `layer.get_weights()`中想要使用正则化的权重的下标。
* `reg_slice`: `reg_index`对应权重的正则化范围，如果都为`None`则整个权重都参与计算，否则只有选中部分参与。
* `reg_factor`: 正则化系数。

### 多头注意力机制

[Transformer](https://arxiv.org/pdf/1706.03762.pdf)中使用的注意力机制，需要指定`head_num`且`head_num`必须要能整除输入的隐藏维度：

```python
from tensorflow import keras
from keras_multi_head import MultiHeadAttention

input_layer = keras.layers.Input(
    shape=(2, 3),
    name='Input',
)
att_layer = MultiHeadAttention(
    head_num=3,
    name='Multi-Head',
)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=att_layer)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics={},
)
model.summary()
```

当输入只有一个tensor时，输入和输出的形状相同；当输入是一个`list`时，会被认为是包含Q、K、V的`list`：

```python
from tensorflow import keras
from keras_multi_head import MultiHeadAttention

input_query = keras.layers.Input(
    shape=(2, 3),
    name='Input-Q',
)
input_key = keras.layers.Input(
    shape=(4, 5),
    name='Input-K',
)
input_value = keras.layers.Input(
    shape=(4, 6),
    name='Input-V',
)
att_layer = MultiHeadAttention(
    head_num=3,
    name='Multi-Head',
)([input_query, input_key, input_value])
model = keras.models.Model(inputs=[input_query, input_key, input_value], outputs=att_layer)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics={},
)
model.summary()
```
