import unittest
import json
from random import shuffle
import numpy as np

num_rows = 75
num_cols = 75

class StatoilDataset:

    def __init__(self, filename, purpose, augmentation_params = None):

        if purpose == "training":
            self._for_training = True
        elif purpose == "testing":
            self._for_training = False
        else:
            raise ValueError('purpose must be either "training" or "testing", but got "{}"'.format(purpose))

        with open(filename, 'rt') as f:
            data_json = json.load(f)

        if self._for_training:
            self._labels = np.array([x['is_iceberg'] for x in data_json])

        self._image_ids = np.array([x['id'] for x in data_json])
        self._band1_images = np.array([x['band_1'] for x in data_json])
        self._band2_images = np.array([x['band_2'] for x in data_json])

    def get_image_from_index(self, index, augmentation = False):
        band1_flat = self._band1_images[index]
        band2_flat = self._band2_images[index]
        both_bands = \
            np.stack([np.reshape(band1_flat, newshape = [num_rows, num_cols]),
                      np.reshape(band2_flat, newshape = [num_rows, num_cols])],
                     axis = 2)
        if augmentation:
            raise NotImplementedError('augmentation not yet implemented')
        return both_bands

    def get_label_from_index(self, index):
        if self._for_training:
            return self._labels[index]
        else:
            raise NotImplementedError('get_label_from_index is not valid when purpose == "testing"')

    @property
    def num_images(self):
        return len(self._image_ids)

class StatoilTrainingDataset:

    def __init__(self, filename='./data/train.json', validation_fraction=0.3, params = None):
        if params is None:
            params = {
                'shuffle' : True
            }

        self._shuffle_examples = params['shuffle']

        self._dataset = StatoilDataset(filename, purpose = "training", augmentation_params = params)

        self._N_total = self._dataset.num_images
        self._N_validation = int(self._N_total*validation_fraction)
        self._N_train = self._N_total - self._N_validation

        if self._shuffle_examples:
            self._train_indices = set(np.random.choice(self._N_total, size = self._N_train, replace = False))
        else:
            self._train_indices = set(np.arange(self._N_train))
        self._reset_training_queue()
        self._validation_indices = list(set(np.arange(self._N_total)) - self._train_indices)

    def _reset_training_queue(self):
        self._train_indices_queue = list(self._train_indices)
        if self._shuffle_examples:
            shuffle(self._train_indices_queue)
        else:
            self._train_indices_queue.sort()

    def get_next_training_indices(self, num_to_get):
        if num_to_get <= len(self._train_indices_queue):
            this_batch = self._train_indices_queue[:num_to_get]
            self._train_indices_queue = self._train_indices_queue[num_to_get:]
            return this_batch
        else:
            this_batch = self._train_indices_queue
            self._reset_training_queue()
            remaining_size = num_to_get - len(this_batch)
            next_batch = self.get_next_training_indices(remaining_size)
            return this_batch + next_batch
            
    def get_validation_indices_as_batches(self, batch_size):
        batches = []
        remaining = self._N_validation
        cursor = 0
        while remaining > 0:
            if remaining > batch_size:
                batches += [self._validation_indices[cursor:cursor+batch_size]]
                remaining -= batch_size
                cursor += batch_size
            else:
                batches += [self._validation_indices[cursor:]]
                remaining = 0
        return batches

    def get_next_training_batch(self, batch_size):
        """Gets the next training batch, performing any data augmentation we
        want.

        Returns:
            (im, lab)
            im is an ndarray of (image_index, row, col, band)
            lab is an ndarray of labels (1 = iceberg, 0 = ship)
        """
        indices = self.get_next_training_indices(batch_size)
        return self.get_batch(indices, augmentation = True)

    def get_batch(self, indices, augmentation = False):
        images = np.array([self._dataset.get_image_from_index(i) for i in indices])
        labels = np.array([self._dataset.get_label_from_index(i) for i in indices])
        return (images, labels)

    def get_validation_set(self, batch_size):
        """Get the entire validation set as batches of the requested size.

        Returns:
            [(im1, lab1), (im2, lab2), ...]
            imx is an ndarray of (image_index, row, col, band)
            labx is an ndarray of labels (1 = iceberg, 0=ship)
            i.e. similar to get_next_training_batch, but over a list
        """
        indices = self.get_validation_indices_as_batches(batch_size)
        return [self.get_batch(i, augmentation = False) for i in indices]

class StatoilDatasetTests(unittest.TestCase):
    
    def test_training_dataset(self):

        train = StatoilTrainingDataset()

        # Check that training and validation indices do not overlap
        self.assertTrue(train._train_indices.isdisjoint(train._validation_indices))

        # Check that we get the whole training set
        ind1 = train.get_next_training_indices(train._N_train // 3)
        ind2 = train.get_next_training_indices(train._N_train // 3)
        ind3 = train.get_next_training_indices(train._N_train - 2*(train._N_train // 3))
        self.assertTrue(set(ind1).isdisjoint(ind2))
        self.assertTrue(set(ind1).isdisjoint(ind3))
        self.assertTrue(set(ind2).isdisjoint(ind3))
        self.assertEqual(set(ind1).union(ind2).union(ind3), train._train_indices)
        self.assertEqual(len(set(ind1).union(ind2).union(ind3)), train._N_train)

        # Check that the next training set is not identical 
        ind1b = train.get_next_training_indices(train._N_train // 3)
        ind2b = train.get_next_training_indices(train._N_train // 3)
        ind3b = train.get_next_training_indices(train._N_train - 2*(train._N_train // 3))
        self.assertNotEqual(ind1b, ind1)
        self.assertNotEqual(ind2b, ind2)
        self.assertNotEqual(ind3b, ind3)
        self.assertTrue(set(ind1b).isdisjoint(ind2b))
        self.assertTrue(set(ind1b).isdisjoint(ind3b))
        self.assertTrue(set(ind2b).isdisjoint(ind3b))
        self.assertEqual(set(ind1b).union(ind2b).union(ind3b), train._train_indices)
        self.assertEqual(len(set(ind1b).union(ind2b).union(ind3b)), train._N_train)

    def test_validation_dataset(self):

        train = StatoilTrainingDataset()

        # Check that we get the validation set
        val = train.get_validation_indices_as_batches(32)
        self.assertEqual(len(val[0]), 32)
        self.assertEqual(len(val[1]), 32)
        self.assertEqual([x for v in val for x in v], train._validation_indices)

        # Check that we get the same thing the second time
        valb = train.get_validation_indices_as_batches(32)
        self.assertEqual(len(valb[0]), 32)
        self.assertEqual(len(valb[1]), 32)
        self.assertEqual(val, valb)

    def test_get_training_batch(self):

        train = StatoilTrainingDataset()

        batch1_im, batch1_lab = train.get_next_training_batch(32)

        self.assertEqual(type(batch1_im), np.ndarray)
        self.assertEqual(type(batch1_lab), np.ndarray)
        self.assertEqual(batch1_im.shape, (32, 75, 75, 2))
        self.assertEqual(batch1_lab.shape, (32,))

    def test_get_validation_set(self):

        train = StatoilTrainingDataset()

        val = train.get_validation_set(32)

        self.assertEqual(val[0][0].shape, (32, 75, 75, 2))
        self.assertEqual(val[0][1].shape, (32,))
