#! /usr/bin/python3

import json, unittest
import numpy as np

num_rows = 75
num_cols = 75

class StatoilDataset:

    def __init__(self, filename, mini_dataset = False):
        
        with open(filename, 'rt') as f:
            data_json = json.load(f)

        if mini_dataset == True:
            data_json = data_json[:100]

        if 'is_iceberg' in data_json[0].keys():
            self._labels = np.array([x['is_iceberg'] for x in data_json])

        self._band1_images = np.array([x['band_1'] for x in data_json])
        self._band2_images = np.array([x['band_2'] for x in data_json])

    def _make_image_zero_mean(image):
        m = np.mean(image)
        return (image - m)

    def _reflect_image_horizontally(image):
        return np.flip(image, axis=1)

    def _reflect_image_vertically(image):
        return np.flip(image, axis=0)

class StatoilTrainingDataset(StatoilDataset):

    def __init__(self, params, filename='./data/train.json', validation_fraction = 0.3, mini_dataset = False):
        super(StatoilTrainingDataset, self).__init__(filename, mini_dataset)

        self._flips = params['flips']
        self._zero_mean_images = params['demean']

        self._N_total = len(self._band1_images)
        self._N_val = int(validation_fraction * self._N_total)
        self._N_train = self._N_total - self._N_val

        self._training_cursor = 0

    def get_image_and_label_from_index(self, validation, index):
        band1_flat = self._band1_images[index]
        band2_flat = self._band2_images[index]
        if self._zero_mean_images:
            band1_flat = StatoilDataset._make_image_zero_mean(self._band1_images[index])
            band2_flat = StatoilDataset._make_image_zero_mean(self._band2_images[index])
        both_bands = \
            np.stack([np.reshape(band1_flat, newshape = [num_rows, num_cols]),
                      np.reshape(band2_flat, newshape = [num_rows, num_cols])],
                     axis = 2)
        label = self._labels[index]
        if self._flips and not validation:
            if np.random.randint(low=0, high=2) == 1:
                both_bands = StatoilDataset._reflect_image_horizontally(both_bands)
            if np.random.randint(low=0, high=2) == 1:
                both_bands = StatoilDataset._reflect_image_vertically(both_bands)
        return both_bands, label

    def get_next_training_image_and_label(self):
        both_bands, label = self.get_image_and_label_from_index(validation=False, index=self._training_cursor)
        self._training_cursor = (self._training_cursor + 1) % self._N_train
        return both_bands, label

    def get_next_training_batch(self, batch_size):
        all_data = [self.get_next_training_image_and_label() for _ in range(batch_size)]
        return (np.array([im for (im, _) in all_data]), np.array([l for (_, l) in all_data]))

    def get_validation_set(self, batch_size):
        validation_cursor = self._N_train
        batches = []
        while validation_cursor < self._N_total:
            num_left = self._N_total - validation_cursor
            num_to_take = batch_size if num_left >= batch_size else num_left
            images_and_labels = [self.get_image_and_label_from_index(validation = True, index=i)
                                 for i in range(validation_cursor, validation_cursor + num_to_take)]
            batches += [(np.array([im for (im, _) in images_and_labels]), np.array([l for (_, l) in images_and_labels]))]
            validation_cursor += num_to_take
        return batches

class DatasetTests(unittest.TestCase):

    _example_params = {'flips' : False,
                       'demean' : True}

    def test_make_image_zero_mean(self):

        im = np.array([[1,2], [3,4], [5,6]])
        im_mz = StatoilDataset._make_image_zero_mean(im)

        self.assertEqual(np.mean(im_mz), 0.0)

    def test_shape_of_training_images(self):

        train = StatoilTrainingDataset(params = DatasetTests._example_params, mini_dataset = True)

        im1, l1 = train.get_next_training_image_and_label()
        self.assertEqual(im1.shape, (75, 75, 2))

        im2, l2 = train.get_next_training_image_and_label()
        self.assertEqual(im2.shape, (75, 75, 2))

        # Check that subsequent calls dont give me identical images
        self.assertTrue((im1 != im2).any())

    def test_shape_of_training_batch(self):

        train = StatoilTrainingDataset(params = DatasetTests._example_params, mini_dataset = True)

        batch_im, batch_labels = train.get_next_training_batch(10)

        self.assertEqual(batch_im.shape, (10, 75, 75, 2))
        self.assertEqual(batch_labels.shape, (10,))

    def test_shape_of_validation_set(self):

        train = StatoilTrainingDataset(params = DatasetTests._example_params, mini_dataset = True)

        validation_set = train.get_validation_set(batch_size = 5)

        # Check we get the right number of labels
        self.assertEqual(train._N_val, sum([len(x[1]) for x in validation_set]))
        # Check that the first set has batch_size labels
        self.assertEqual(5, len(validation_set[0][1]))
