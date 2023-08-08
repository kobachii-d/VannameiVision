<p align="justify">
    <h1>VannameiVision</h1>
</p>

<p align="justify">
In aquaculture, early detection of susceptible shrimp larvae is of paramount importance to maintain healthy production environments. This GitHub repository introduces VannameiVision, a novel approach that combines probabilistic deep learning with transfer and deep metric learning techniques to address the challenge of accurately identifying shrimp larvae in such vulnerable conditions.
</p>

<img src="architecture.jpg" alt="Architecture of VannameiVision Model" style="max-width:30%;">

<p align="justify">
    <h2>Dependencies</h1>
</p>

```
numpy==1.22.4
scikit-image==0.19.3
tensorflow==2.12.0
tensorflow-addons==0.21.0
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
