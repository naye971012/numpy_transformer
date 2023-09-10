import torch

# 예제로 사용할 Torch 텐서를 생성합니다.
tensor = torch.tensor([0.2, 0.5, 1.2, -0.1, 0.8])

# 0과 1 사이의 값을 가지지 않는 요소를 찾습니다.
out_of_range_elements = torch.logical_or(tensor < 0, tensor > 1)

# 결과를 출력합니다.
print(out_of_range_elements)