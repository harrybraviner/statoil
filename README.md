# Dataset notes

 * Training dataset contains 1604 images, 753 of them (46.9%) are icebergs.
 * Can I augment these (e.g. relections) usefully?
 * Can I do translations and rotations on training dataset? Could if I was prepared to 'zero' much of the background.
 * In almost all cases (99.7% of training data, all but 5 images), band 1 has a brighter maximum than band 2.
 
# Design notes

* Small dataset, regularization really matters! Use dropout and dataset augmentation.

# StatNet1

Convolutional neural net.
Two concolutional layers, then two fully connected layers.
Non-linearities will be ReLU.
Dropout to be used for training in fully connected layers.

# ToDo

## Programming

* Add noise augmentation of data.
* Check that your conv layers are able to 'mix' the two bands!

## Experimentation

* Does the exponentiation help?
* Do the flips improve things?
* Does noise augmentation improve things?
* Does dropout improve things?
* Start a 'lab-book' document.
* Where am I going to save results to?
