from PIL import Image
from matplotlib import pyplot
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import make_grid

# Generate a sample gene for everyone to understand
FILE_NAME = '../img/test/girl.png'

img = Image.open(FILE_NAME)  # Open file
print(type(img))
img.thumbnail((400, 400))  # Resize image and keep ratio

# Gene type: pytorch.tensor
gene1 = transforms.PILToTensor()(img).permute(1, 2, 0)
print(type(gene1))
print(gene1.size())

print()

# Gene type: numpy.ndarray
gene2 = np.asarray(img)
print(type(gene2))
print(gene2.shape)

# Show image by pyplot
pyplot.figure(num="Test")
pyplot.axis("off")
pyplot.imshow(gene1)
# pyplot.ion()
pyplot.show()

# # Directly show the image
# img2.show()
