# Dataset notes

 * Training dataset contains 1604 images, 753 of them (46.9%) are icebergs.
 * Can I augment these (e.g. relections) usefully?
 * Can I do translations and rotations on training dataset? Could if I was prepared to 'zero' much of the background.
 
# Design notes

* Small dataset, regularization really matters! Use dropout.

# StatNet1

Convolutional neural net.
Two concolutional layers, then two fully connected layers.
Non-linearities will be ReLU.
Dropout to be used for training in fully connected layers.

# ToDo

* Record training performance in a file.
* Record parameters used to train the net.
* Add data-augmentation to training.
* Add dropout into the net.
