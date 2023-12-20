## **Week 5 - Normalization/Residual Block**
- **Topic**
  - LayerNorm
  - BatchNorm
- **Week 5 Exercise**
  - compare w/wo Normalization Layers in terms of converge speed/accuracy

</br>

## Exercise Description
- in this exercise, we will check the effect of the normalization
- using fashion-mnist dataset, mlp model
- fill blank in batchnorm/layernorm.py
- compare 3 models in terms of converge speed and accuracy
- 1. w/o normalization, 
- 2. w/ batchnorm, 
- 3. w/ layernorm

</br>

## Excepted output
- training model with very high LR(0.1)

</br>

- model without normalization
![week_5_output_1](https://github.com/naye971012/numpy_transformer/assets/74105909/caf39415-f93a-4415-917c-3e297565329f)

- model with layer normalization
![week_5_output_2](https://github.com/naye971012/numpy_transformer/assets/74105909/e1be1942-5895-47ee-a2eb-0e8ffed34b09)

- model with batch normalization
![week_5_output_3](https://github.com/naye971012/numpy_transformer/assets/74105909/50ebeeb2-a187-49ea-9f3a-f38c23691605)

- with normalization, model converge fast and well even learning rate is very high(0.1)