# Comprint
Comprint is an image forgery detection and localization method that utilizes compression fingerprints.

## License
These files were created by IDLab-MEDIA, Ghent University - imec, in collaboration with the Image Processing Research Group of the University Federico II of Naples (GRIP-UNINA).

All rights reserved.

This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the terms of the license, as specified in the document LICENSE.txt (included in this package).

IDLab-MEDIA: https://media.idlab.ugent.be/

GRIP-UNINA: https://www.grip.unina.it/

## Installation
The code requires Python 3.X and was built with Tensorflow 2.9.1.

Install the requested libraries using:
```
pip install -r code/requirements.txt
```

## Usage
### Training
First, download the training and validation data, and place it in data/train and data/validation, respectively.
Downloadlinks can be found in data/downloadlinks_train_and_validation.txt.

Training settings can be changed with the corresponding variables in code/train_network.py and train_network_siamese.py.

Then the shell scripts in the main folder start the training. For training the pre-trained network that estimates JPEG artifacts:

```
bash run-training.sh
```

For training the siamese network that extracts the comprint:
```
bash run-training-siamese.sh
```

### Comprint and heatmap extraction
The Jupyter notebook code/get_comprint_heatmap.ipynb gives an example on how to extract the comprint and heatmap. By changing the filename / path, you can extract the comprint from other images under investigation. Our trained models are included in the models folder.

## Reference
This work will be presented in the [Workshop on MultiMedia FORensics in the WILD (MMFORWILD) 2022](https://iplab.dmi.unict.it/mmforwild22/), held in conjunction with the [International Conference on Pattern Recognition (ICPR) 2022](https://www.icpr2022.com/).

```js
@inproceedings{mareen2022comprint,
  title={Comprint: Image Forgery Detection and Localization using Compression Fingerprints},
  author={H. Mareen, D. Vanden Bussche, F. Guillaro, D. Cozzolino, G. Van Wallendael, P. Lambert, L. Verdoliva,
  booktitle={Proceedings of the International Conference on Pattern Recognition},
  year={2022},
  organization={Springer}
} 
```
