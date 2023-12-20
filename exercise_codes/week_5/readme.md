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
![week_5_w/o_norm](../../images/week_5_output_1.png)

- model with layer normalization
![week_4_w/_layernorm](../../images/week_5_output_2.png)

- model with batch normalization
![week_4_w/_batchnorm](../../images/week_5_output_3.png)

- with normalization, model converge fast and well even learning rate is very high(0.1)