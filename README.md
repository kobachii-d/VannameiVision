<p align="justify">
    <h1>VannameiVision</h1>
</p>

<p align="justify">
In aquaculture, early detection of susceptible shrimp larvae is of paramount importance to maintain healthy production environments. This repository introduces VannameiVision, a novel approach that combines probabilistic deep learning with transfer and deep metric learning techniques to address the challenge of accurately identifying shrimp larvae in such vulnerable conditions.
</p>

<img src="architecture.jpg" alt="Architecture of VannameiVision Model" style="max-width:30%;">

<p align="justify">
    <h2>Installation</h1>
</p>

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
