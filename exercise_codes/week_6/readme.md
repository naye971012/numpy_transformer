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

![week_6_w/o_mask](../../images/week_6_output_1.png)

- model with mask example

![week_6_w/_mask](../../images/week_6_output_2.png)