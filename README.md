# numpy_transformer
numpy implementation of pytorch deep learning models including Transformer \
6 weeks basic study curriculum with exercise code

2023-2 인공지능 연합동아리 프로메테우스 4기 강의/실습 자료
[prometheus-ai website](https://prometheus-ai.net/about)

<details>
    <summary> Studying Curriculum </summary>

## **Week 1 - Basis**
- **Topic** 
  - Activation Functions (ReLU, Sigmoid)
  - Loss Functions (Cross-Entropy, Binary Cross-Entropy)
  - Linear Layer
- **Week 1 Exercise**
  - Compare NumPy implementation with Torch functions

## **Week 2 - CNN**
- **Topic**
  - Convolution Layer (2D)
  - MaxPooling (2D)
  - Flatten Layer
- **Week 2 Exercise**
  - Compare NumPy/Torch model accuracy with Linear/CNN models

## **Week 3 - Optimizer/Embedding**
- **Topic**
  - Dropout
  - Embedding
  - Positional Encoding
  - Optimizers (SGD, momentum, Adam)
- **Week 3 Exercise**
  - Compare NumPy model accuracy with different Optimizers

## **Week 4 - RNN/Tokenizer**
- **Topic**
  - RNN
  - Word Tokenzizer
- **Week 4 Exercise**
  - Train an RNN model using a word tokenizer and monitor only the loss decline.

## **Week 5 - Normalization/Residual Block**
- **Topic**
  - LayerNorm
  - BatchNorm
- **Week 5 Exercise**
  - compare w/wo Normalization Layers in terms of converge speed/accuracy

## **Week 6 - Attention**
- **Topic**
  - Attention
- **Week 6 Exercise**
  - Visualize Attention layer
  - Visualize Masked Attention map

</details>

</br>

<details>
  <summary> Exercise Example </summary>

  ![exercise_example](https://github.com/naye971012/numpy_transformer/assets/74105909/d78da2df-b736-4978-84ed-c4fc68dd9815)
</details>

</br>

<details>
  <summary> Model Training Example </summary>

### 1. **BERT (Transformer-Encoder archicture)** <a name="bert"></a>
- I've trained on the "Abirate/english_quotes" dataset, containing around 3000 sentences.
- Example data:
```
'“Be yourself; everyone else is already taken.”' 
```

</br>

- Experiment step
```python
    1. Preprocessed using a word tokenizer.
    2. Trained a numpy-based BERT model.
    3. Trained a torch-based BERT model.
    4. Compared the two models based on their train/validation loss decline.
```

</br>

- Training Arguments ( model size: 9M )
```yaml
    Training Arguments:
    - Batch Size: 6
    - Learning Rate: 0.001
    - Epochs: 20

    BERT Model Arguments:
    - Number of Blocks: 6
    - Maximum Sentence Length: 50
    - Embedding Layer Dimension: 256
    - Number of Heads for Self-Attention: 4
```

</br>

- **Since training a language model on a CPU takes a considerable amount of time, I monitored the loss decline without full training.**

- Bert implemented by torch
![th_bert](https://github.com/naye971012/numpy_transformer/assets/74105909/5b9351d3-7b77-4621-9cff-680a5a92cfb4)

- Bert implemented by numpy
![np_bert](https://github.com/naye971012/numpy_transformer/assets/74105909/41913c50-031e-47bc-8162-b974710b5c88)

</details>

## Models
- Bert (Transformer-Encoder Architecture)
- GAN (Generative Adversarial Network)
- Transformer

## Completed Checklist

Here is the current checklist of completed items:

### Activation Functions
- ReLU
- Sigmoid
- Softmax
- Tanh

### Loss Functions
- Cross-Entropy Loss
- Binary Cross-Entropy Loss

### Layers
- Linear Layer
- CNN (Convolutional Neural Network)
- RNN (Recurrent Neural Network)
- Scaled Dot Product Attention

### Optimizers
- SGD
- SGD with momentum
- Adam

### Utilities
- Dropout
- Embedding
- Positional Encoding
- Flatten
- MaxPooling

### Tokenizer
- Vocabulary set
- Word Tokenizer

### Normalization
- Batch Normalization
- Layer Normalization

### Blocks
- Plain Residual Block



## Tree
```
.
├── exercise_studying_functions
│   ├── readme.md
│   ├── week_1
│   │   ├── codes
│   │   │   ├── binary_ce.py
│   │   │   ├── linear.py
│   │   │   ├── relu.py
│   │   │   └── sigmoid.py
│   │   ├── readme.md
│   │   └── week_1_relu-sigmoid-bceloss-mlp.py
│   ├── week_2
│   │   ├── codes
│   │   │   ├── cnn.py
│   │   │   ├── flatten.py
│   │   │   └── pooling.py
│   │   ├── readme.md
│   │   ├── week_2_cnn_main.py
│   │   └── week_2_models.py
│   ├── week_3
│   │   ├── codes
│   │   │   ├── Adam.py
│   │   │   ├── dropout.py
│   │   │   ├── embedding.py
│   │   │   ├── positional_encoding.py
│   │   │   ├── SGD_momentum.py
│   │   │   └── SGD.py
│   │   ├── readme.md
│   │   ├── week_3_models.py
│   │   └── week_3_optimizer_main.py
│   ├── week_4
│   │   ├── codes
│   │   │   ├── rnn.py
│   │   │   ├── vocab.py
│   │   │   └── word_tokenizer.py
│   │   ├── data
│   │   ├── readme.md
│   │   ├── week_4_models.py
│   │   └── week_4_rnn_tokenizer_main.py
│   ├── week_5
│   │   ├── codes
│   │   │   ├── batchnorm.py
│   │   │   └── layernorm.py
│   │   ├── readme.md
│   │   ├── week_5_models.py
│   │   └── week_5_normalization.py
│   └── week_6
│       ├── codes
│       │   └── attention.py
│       ├── readme.md
│       ├── week_6_attention.py
│       └── week_6_models.py
├── exercise_training_model
│   ├── numpy_ver
│   │   ├── BERT(transformer-encoder)
│   │   │   ├── bert.py
│   │   │   ├── dataset.py
│   │   │   └── main.py
│   │   ├── GAN
│   │   │   ├── dataset.py
│   │   │   ├── main.py
│   │   │   └── output_example
│   │   └── Transformer
│   │       └── transformer.py
│   ├── readme.md
│   └── torch_ver
│       ├── BERT(transformer-encoder)
│       │   ├── bert.py
│       │   ├── dataset.py
│       │   └── main.py
│       └── GAN
│           └── readme.md
├── numpy_functions
│   ├── activations
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── softmax.py
│   │   └── tanh.py
│   ├── blocks
│   │   └── residual_block.py
│   ├── commons
│   │   ├── attention.py
│   │   ├── cnn.py
│   │   ├── linear.py
│   │   ├── multi_head_attention.py
│   │   └── rnn.py
│   ├── __init__.py
│   ├── losses
│   │   ├── binary_ce.py
│   │   └── ce.py
│   ├── metric
│   │   └── metric.py
│   ├── normalization
│   │   ├── batchnorm.py
│   │   └── layernorm.py
│   ├── optimizer
│   │   ├── Adam.py
│   │   ├── SGD_momentum.py
│   │   └── SGD.py
│   ├── tokenizer
│   │   ├── vocab.py
│   │   └── word_tokenizer.py
│   └── utils
│       ├── dropout.py
│       ├── embedding.py
│       ├── flatten.py
│       ├── pooling.py
│       └── positional_encoding.py
├── numpy_models
│   ├── base_module.py
│   ├── gan
│   │   ├── __init__.py
│   │   └── simple_gan.py
│   ├── __init__.py
│   ├── readme.md
│   └── transformer
│       ├── decoder_block.py
│       ├── decoder.py
│       ├── encoder_block.py
│       ├── encoder.py
│       ├── feedforward.py
│       ├── __init__.py
│       └── transformer.py
└── README.md
```