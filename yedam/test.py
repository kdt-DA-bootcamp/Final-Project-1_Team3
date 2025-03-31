import torch
print(torch.cuda.is_available())  # → True가 되어야 정상
print(torch.cuda.get_device_name())  # → GPU 이름 출력