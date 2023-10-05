# numpy_transformer
numpy implementation of deep learning models including Transformer


```

.
├── dummy_codes.py
├── README.md
├── exercise_codes
│   ├── week_1_relu-sigmoid-bceloss-mlp.py
│   ├── week_2_cnn_main.py
│   ├── week_2_models.py
│   ├── week_3_models.py
│   └── week_3_optimizer_main.py
├── numpy_models
│   ├── activations
│   │   ├── relu.py
│   │   └── sigmoid.py
│   ├── commons
│   │   ├── cnn.py
│   │   └── linear.py
│   ├── losses
│   │   ├── binary_ce.py
│   │   └── ce.py
│   ├── optimizer
│   │   ├── Adam.py
│   │   ├── SGD_momentum.py
│   │   └── SGD.py
│   └── utils
│       ├── dropout.py
│       ├── embedding.py
│       ├── pooling.py
│       ├── positional_encoding.py
│       └── flatten.py
└── torch_models
    ├── activations
    │   ├── relu.py
    │   └── sigmoid.py
    ├── commons
    │   ├── cnn.py
    │   └── linear.py
    ├── losses
    │   ├── binary_ce.py
    │   └── ce.py
    ├── optimizer
    │   ├── Adam.py
    │   ├── SGD_momentum.py
    │   └── SGD.py
    └── utils
        ├── dropout.py
        ├── embedding.py
        ├── pooling.py
        ├── positional_encoding.py
        └── flatten.py
```


# Completed Checklist

Here is the current checklist of completed items:

## Activation Functions
- ReLU (Rectified Linear Unit)
- Sigmoid

## Loss Functions
- Cross-Entropy Loss
- Binary Cross-Entropy Loss

## Layers
- Linear Layer
- CNN (Convolutional Neural Network)

## Optimizers
- SGD
- SGD with momentum
- Adam

## Utilities
- Dropout
- Embedding
- Positional Encoding
- Flatten
- MaxPooling



# TODO LIST

## Layers
- Attention 
- RNN 
- GRU

## Utilities
- Layer Normalization 
- Batch Normalization 

## Tokenizers
- word tokenizer 

## Dataset / Dataloader
- Dataset


# Curriculum
## **Week 1**
- **Topic**
  - Activation Functions (ReLU, Sigmoid)
  - Loss Functions (Cross-Entropy, Binary Cross-Entropy)
  - Linear Layer
- **Week 1 Exercise**
  - Compare NumPy implementation with Torch functions

## **Week 2**
- **Topic**
  - Convolution Layer (2D)
  - MaxPooling (2D)
  - Flatten Layer
- **Week 2 Exercise**
  - Compare NumPy/Torch model accuracy with Linear/CNN models

## **Week 3**
- **Topic**
  - Dropout
  - Embedding
  - Positional Encoding
  - Optimizers (SGD, momentum, Adam)
- **Week 3 Exercise**
  - Compare NumPy model accuracy with different Optimizers