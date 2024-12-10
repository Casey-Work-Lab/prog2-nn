import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

ds_train = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True,
)

print(f"numbers of datasets:{len(ds_train)}")

image, target = ds_train[1]
print(type(image), target)

plt.imshow(image, cmap="gray_r")
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, dtype=torch.float32, scale=True)
print(image.shape, image.dtype)
print(image.min(), image.max())



# for i in range (5):
#    for j in range(5):
#        k = i*5+j
#        image, target = ds_train[k]
#        plt.subplot(5, 5, i*5+j+1)
#        plt.imshow(image, cmap="gray_r")
#        plt.title(target)
# plt.show()

# ctrl + / 