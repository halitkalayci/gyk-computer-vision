import torch
import torch.version

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

a = torch.rand(3,3).cuda()
print(a)