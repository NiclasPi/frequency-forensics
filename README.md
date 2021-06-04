<!--
<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>
-->

# Wavelet-Packet Powered Deepfake Image Detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the supplementary source code for our paper
Wavelet-Packet Powered Deepfake Image Detection,
which is currently under review.


## Installation
The latest code can be installed in development mode with:
```shell
$ pip install -e .
```

## Data sets:
We utilize three datasets which commonly appeard in previous work:
-  [FFHQ](https://github.com/NVlabs/ffhq-dataset)
-  [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
-  [LSUN bedroom](https://github.com/fyu/lsun)

## Gan Architectures:
-  [StyleGan](https://github.com/NVlabs/stylegan)
...
TODO add more.

## Getting the ffhq experiment to run:
Download the 128x128 pixel version of the ffhq data sets.
Insert a for loop a random seed and code to resize i.e. 
``` PIL.Image.fromarray(images[0], 'RGB').resize((128, 128)).save(png_filename)```
into 
[ffhq-stylegan](https://github.com/NVlabs/stylegan/blob/03563d18a0cf8d67d897cc61e44479267968716b/pretrained_example.py)
generate 70k images .
Store all images in a `data` folder. Use i.e
```
./data/source_data/A_ffhq
./data/source_data/B_stylegan
```

Afterwards run:
```shell
$ python -m freqdect.prepare_dataset_batched ./data/source_data/ --packets
$ python -m freqdect.prepare_dataset_batched ./data/source_data/
```
now you should be able to train a classifier using
```shell
$ python -m freqdect.train_classifier --data-prefix ./data/source_data/ --calc-normalization
```
