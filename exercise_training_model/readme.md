## Completed List:

1. BERT (Transformer-Encoder archicture) [Bert](#bert) 
2. GAN (Generative Adversarial Network) [GAN](#gan)

</br>

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

</br>
</br>
</br>

2. **GAN (Generative Adversarial Network)** <a name="gan"></a>
- currently, it has 'mode collapse' problem
- I implemented GAN/Diffusion in another repository, [Generative Models](https://github.com/naye971012/Generative_models)