"""
| Store global variables for other files
"""
import torchvision.transforms as T

device = "cpu"
transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
