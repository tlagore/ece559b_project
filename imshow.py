import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

img = Image.open('snake.png').convert("RGB")
print(img.mode)
numpydata = np.asarray(img)

# df = pd.read_csv('state.csv')
# plt.imshow(df)
# plt.savefig('snake.png')
# plt.show()

print(numpydata.shape)
print(numpydata)