from PIL import Image
from matplotlib import pyplot
import numpy as np


# Generate a sample gene for everyone to understand
FILE_NAME = '../img/test/girl.png'

img = Image.open(FILE_NAME)  # Open file
print(type(img))
img.thumbnail((400, 400))  # Resize image and keep ratio

gene = np.asarray(img)  # Convert image to numpy array
print(type(gene))
print(gene)

# Show image by pyplot
pyplot.figure(num="Test")
pyplot.axis("off")
pyplot.imshow(gene)
# pyplot.ion()
pyplot.show()

print(3)
# # Directly show the image
# img.show()
