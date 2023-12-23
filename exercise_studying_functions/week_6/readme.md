## **Week 6 - Attention**
- **Topic**
  - Attention
- **Week 6 Exercise**
  - Visualize Attention layer
  - Visualize Masked Attention map

</br>

## Exercise Description
- in this exercise, we will train attention model
- using custom dataset
```python
data = np.arange(10)
np.random.shuffle(data)
label = np.array([ i+1 if data[i]==i else 0 for i in range(len(data)) ])
```
- visualize attention map


</br>

## Excepted output

- **but it doesn't work well now** (waiting for PR...)

- model without mask example

![week_6_output_1](https://github.com/naye971012/numpy_transformer/assets/74105909/22de0ac7-f63c-44b1-9578-9d795bd8a1b1)

- model with mask example

![week_6_output_2](https://github.com/naye971012/numpy_transformer/assets/74105909/54556d53-2feb-42a4-a946-2f7057f84fad)
