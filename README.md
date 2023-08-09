<p align="justify">
    <h1>VannameiVision</h1>
</p>

<p align="justify">
In aquaculture, early detection of susceptible shrimp larvae is of paramount importance to maintain healthy production environments. This repository introduces VannameiVision, a novel approach that combines probabilistic deep learning with transfer and deep metric learning techniques to address the challenge of accurately identifying shrimp larvae in such vulnerable conditions.
</p>

<img src="www/architecture.jpg" style="height: 500px;">

## Features

### Example data

<p align="justify">
We provide example data of robust and susceptible shrimp larvae (5 each).
</p>

```
from skimage import io
from vannameivision import *

import matplotlib.pyplot as plt

path = sorted(get_image_paths())

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(io.imread(path[0]))
ax1.set_title(path[0].split("/")[-1])
ax1.axis("off")
ax2.imshow(io.imread(path[5]))
ax2.set_title(path[5].split("/")[-1])
ax2.axis("off")
```

<img src="www/example_data.jpg" style="height: 200px;">

### Read and preprocessing

<p align="justify">
    After reading the image, we apply these steps:
    <ul>
        <li>Log-scale adjustment</li>
        <li>Pad symmetrically to form a square</li>
        <li>Resize the image to 224x224</li>
        <li>Augment by random rotation between -360 and 360 degrees</li>
    </ul>
</p>

```
from skimage import io
from vannameivision import *

import matplotlib.pyplot as plt

path = sorted(get_image_paths())
path = path[0]

original = io.imread(path)

preprocessed = read_preprocess(path, augment=15)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(original)
ax1.set_title("Original")
ax1.axis("off")
ax2.imshow(preprocessed[0])
ax2.set_title("Example 1")
ax2.axis("off")
ax3.imshow(preprocessed[1])
ax3.set_title("Example 2")
ax3.axis("off")
ax4.imshow(preprocessed[2])
ax4.set_title("Example 3")
ax4.axis("off")
```

<img src="www/original_vs_preprocessed.jpg" style="height: 200px;">

<p align="justify">
Run this code to install:
</p>

```
pip install git+https://github.com/kobachii-d/VannameiVision.git
```

<p align="justify">
    <h2>Usage</h1>
</p>

<p align="justify">
To predict whether an image contains robust or susceptible shrimp larvae:
</p>

```
python main.py [path_to_image]
```

<p align="justify">
This command takes in the path to the image you wish to analyze. After processing, it will display a prediction along with a confidence percentage.
</p>

<p align="justify">
    <h2>Examples</h1>
</p>

<p align="justify">
    <h3>Robust shrimp larvae</h1>
</p>

<p align="justify">
Input:
</p>

<img src="image/robust/R2.jpg" alt="Robust shrimp larvae" style="width: 150px;">

```
python main.py images/image/robust/R1.jpg
```

<p align="justify">
Output:
</p>

```
Prediction: Robust
Confidence: 92.3%
```

<p align="justify">
    <h3>Susceptible shrimp larvae</h1>
</p>

<p align="justify">
Input:
</p>

<img src="image/susceptible/S5.jpg" alt="Susceptible shrimp larvae" style="width: 150px;">

```
python main.py images/image/susceptible/S1.jpg
```

<p align="justify">
Output:
</p>

```
Prediction: Susceptible
Confidence: 100.0%
```

<p align="justify">
    <h2>Citation</h1>
</p>

TBA

<p align="justify">
    <h2>Acknowledgements</h1>
</p>

<p align="justify">
We sincerely thank the <a href="https://www.biotec.or.th/" target="_blank">National Center for Genetic Engineering and Biotechnology (BIOTEC)</a>, <a href="https://pccp.ac.th/" target="_blank">Princess Chulabhorn Science High School Pathum Thani (PCSHS)</a>, and our families for their support and encouragement.
</p>
