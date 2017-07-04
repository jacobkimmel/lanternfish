# Lanternfish

![Lanternfish logo](lanternfish_logo.png)

`Lanternfish` is a set of software tools to analyze motion data with convolutional neural networks (CNNs). `Lanternfish` converts recorded motion paths in 2-dimensions into 3-dimensional 'motion cube' images, representing motion in an entirely spatial manner. `Lanternfish` includes CNN architectures suitable for classification of these 3-dimensional 'motion cubes' or unsupervised learning of motion features by autoencoding.

## Lanternfish Core Features

### Conversion of motion paths into 3-dimensional images

2-dimensional motion paths are converted into 3-dimensional images with dimensions `(x, y, time)`. Each timepoint in the series is represented by a slice of the resulting 'motion cube.' Each slice marks the location of the object at the corresponding timepoint as the center of an anisotropic kernel.

A typical kernel may be a Gaussian with a broad `sigma` and a unit depth in the time domain. The magnitude `mu` of the kernel may be specified independently at each timepoint to allow for an additional information parameter to be encoded within the motion cube. Encoding the instantaneous speed of an object as the `mu` parameter tends to be useful for general classification tasks. `Lanternfish` also contains tools to specify a truly dynamic kernel for each timepoint, for instance encoding an additional parameter as `sigma`, by convolution on either CPUs or CUDA capable GPUs.

Motion cube generation tools also include the option to compress or crop collected tracks. This feature is useful to deal with limited GPU memory for downstream CNN training. Compression is performed by simple division and rounding of path coordinates, reducing the number of pixels required to represent the full field-of-view in each slice of a motion cube. Cropping allows for removal of a minority of paths that require much larger fields of view to fit, preventing a few outliers from 'diluting' the other motion cubes with empty space.

### Cell Mimetic Simulations of Motion and Transfer Learning

`Lanternfish` contains tools to simulate motion that mimics a sample of heterogeneous motion phenotypes, referred to as "cell mimesis". Sample motility behaviors are mimicked by decomposing the observed behavior into a set of *k* clusters based on displacement and directionality features, then simulating each of these clusters by fitting a Johnson distribution to displacement and turn angle observations within the cluster. Simulations are generated from each cluster proportional to their representation in the original sample.

### Motion Classification CNNs

CNN architectures are included which have proven effective in classification of different types of motion. Architectures optimized for different motion cube sizes are provided.  

### Motion Autoencoding CNNs

CNN architectures for autoencoding of motion cubes to learn representations of motion feature space in an unsupervised manner are included. Architectures optimized for different motion cube sizes are provided.
