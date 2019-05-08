#!/bin/python3

import h5py
import os
import functools

import cv2 as cv
import numpy as np

'''
AutoBatcher is a tiny python3 module for assisting deep learning
trainings on computer vision tasks. It prevents the main training file
to turn into some chaotic spagetti (which we all suffer from.)


Some remarks,
- It automatically decides on the loading mode: "in-memory training" or
	"loading serialized data from disk" according to the available memory
	and dataset size.
- Supports two modes of data preparation: from a "well-organized directory
	structure" or a meta-data file that contains the path and category
	information for each image.
- Uses h5py for serialization/deserialization of data.
- Leverages numpy for fast array indexing/sorting/accessing.
- Uses opencv to load images.
- Accepts preprocessed square images, sizes of which are known and same.


Development roadmap includes,
- Preprocessing and augmentation methods on arbitrarily-collected data.
- PIL support for those who don't like opencv.
- Online batch resampling support for imbalanced datasets.
- Windows support.
'''

np.random.seed(1)

class AutoBatcher:
	def __init__(self,):
		# Config for training
		self._minibatch_size = 0
		self._dataset_name = ""

		# Actual Data
		self._data_tensor = None
		self._label_list = None
		self._data_filenames_list = []
		self._f = None

		# Dynamic elements that change during training
		self._dataset_order_indices= None
		self._current_batch_idx = 0
		self._loss_list = None

		# Data dimensions
		self._width = 0
		self._height = 0
		self._channels = 0
		self._no_of_classes = 0
		self._no_of_samples = 0


	@classmethod
	def loadDatasetFromPath(cls,
													dataset_basedir_path,
													img_size,
													metafile_path="",
													metafile_seperator="",
													memory_fraction=1.0):
		if not os.path.isdir(dataset_basedir_path):
			raise AssertionError("[AutoBatcher] Given dataset path is not valid.")
		elif str(dataset_basedir_path).endswith("/"):
			dataset_basedir_path = dataset_basedir_path[0:-1]

		if not str(img_size).isdigit():
			raise AssertionError("[AutoBatcher] Given image size is not valid.")

		ti = cls()

		dataset_path_contains_all_dirs = functools.reduce(lambda x,y: x and y,
																											[ os.path.isdir(os.path.join(dataset_basedir_path,i))
																												for i in sorted(os.listdir(dataset_basedir_path)) ]
																											)
		data_filenames, label_list = [], []


		if dataset_path_contains_all_dirs:
			print('[AutoBatcher] Loading from dirs.')
			category_names_as_dirs = sorted(os.listdir(dataset_basedir_path))

			category = 0
			for root, dirs, files in os.walk(dataset_basedir_path):
				if len(files) == 0: continue
				dirs.sort()
				onehot = len(category_names_as_dirs)*[0.0]
				onehot[category] = 1.0
				for idx, filename in enumerate(files):
					data_filenames.append( os.path.join(root, filename) )
					label_list.append( onehot )
				category += 1
			ti._no_of_classes = len(category_names_as_dirs)

		else:
			print('[AutoBatcher] Loading from metafile.')
			if metafile_path == "":
				raise AssertionError("[AutoBatcher] Parameter \"metafile_path\" of loadDatasetFromPath() cannot "
														 "be empty when base dataset path contains entries that are not directories.")

			metafile_rawstr = ""
			with open(metafile_path) as f:
				metafile_rawstr = f.read()
			metafile_lines = metafile_rawstr.split('\n')

			sep = None
			if metafile_seperator == "":
				seps = {"\t":0, ",":0, " ":0}
				for dataline in metafile_lines:
					for sep in seps:
						if sep in dataline:
							seps[sep] += 1
				sep = seps[ max( {v:k for k,v in seps.items()} ) ]
			else:
				sep = metafile_seperator
			
			for dataline in metafile_lines:
				img_path, label = dataline.split(sep)
				data_filenames.append( os.path.join(dataset_basedir_path, img_path) )
				label_list.append( label )

			category_mapping = {v:k for k,v in enumerate(set(label_list))}

			ti._no_of_classes = len(category_mapping)
			for idx, lbl in enumerate(label_list):
				onehot = len(category_mapping)*[0.0]
				onehot[ category_mapping[lbl] ] = 1.0
				label_list[idx] = onehot

		ti._dataset_name = dataset_basedir_path[ dataset_basedir_path.rfind("/")+1 :]

		ti._data_filenames_list = data_filenames
		ti._no_of_samples = len(data_filenames)
		ti._dataset_order_indices = np.arange( ti._no_of_samples )

		ti._width = img_size
		ti._height = img_size
		ti._channels = 3

		datatensor = None
		labels = None
		total_vars = ti._no_of_samples*ti._width*ti._height*ti._channels
		
		if ti._inMemoryTrainingIsPossible(memory_fraction, total_vars):
			datatensor = np.ndarray(shape=(ti._no_of_samples, ti._height, ti._width, ti._channels), dtype=np.float32)
			labels 		 = np.ndarray(shape=(ti._no_of_samples, ti._no_of_classes ), dtype=np.float32)
		else:
			decision = input('[AutoBatcher] Dataset is too large to fit in memory. Serialize to hdf5 file? [Y/n]\n')
			if str(decision).lower() != 'y':
				print("[AutoBatcher] Exiting.")
				exit()
			else:
				h5_dir_path = input('Please provide a path for serialized file.\n')
				ti._f = h5py.File( os.path.join(h5_dir_path,ti._dataset_name)+(".hdf5"), "w")
				datatensor = ti._f.create_dataset(ti._dataset_name+"_images", (ti._no_of_samples, ti._height,ti._width,ti._channels), dtype='f')
				labels 		 = ti._f.create_dataset(ti._dataset_name+"_labels", (ti._no_of_samples, ti._no_of_classes ), 	dtype='f')
			
		for idx, filepath in enumerate(data_filenames):
			datatensor[idx,:,:,:] = 1.0-(cv.imread(os.path.join(dataset_basedir_path, filepath))[0:ti._height,0:ti._width,:] / 255.0)
		
		for idx, label in enumerate(label_list):
			labels[idx] = label

		ti._data_tensor = datatensor
		ti._label_list = labels

		return ti


	@classmethod
	def loadDatasetFromSerializedFile(cls, serialized_file_full_path):
		if not os.path.isfile(serialized_file_full_path):
			raise AssertionError('[AutoBatcher] Serialized file cannot be found in given path.')
		ti = cls()

		ti._f = h5py.File( serialized_file_full_path, "r")
		[a,b] = list(ti._f.keys())
		
		# Get all data
		ti._no_of_samples = len(ti._f[a])

		probed_image = ti._f[a][0,:,:,:]
		[ti._height, ti._width, ti._channels ] = list(probed_image.shape)

		print('[AutoBatcher] Loaded image dims: {} {} {}'.format(ti._height, ti._width, ti._channels))

		probed_label = ti._f[b][0,:]
		[ti._no_of_classes] = list(probed_label.shape)

		ti._dataset_name = a[:str(a).find("_images")]
		ti._data_tensor	= ti._f[a]
		ti._label_list 	= ti._f[b]
		ti._dataset_order_indices = np.arange( ti._no_of_samples )
		ti._data_filenames_list 	= []

		return ti


	def serializeToDisk(self,h5_filepath,dset_name,dim_dict):
		f = h5py.File( os.path.join(h5_filepath,dset_name)+(".hdf5"), "w")
		f.create_dataset(dset_name+"_images", (dim_dict["b"], dim_dict["h"],dim_dict["w"],dim_dict["c"]), dtype='f')
		f.create_dataset(dset_name+"_labels", (dim_dict["b"], dim_dict["nc"] ), 	dtype='f')
		f.close()


	def setMiniBatchSize(self,size):
		self._minibatch_size = size


	def getNextMiniBatch(self, sampling_policy='random'):
		minibatch_datatensor = None
		minibatch_labels 		 = None
		if sampling_policy == 'random':
			src  = self._current_batch_idx
			trgt = self._current_batch_idx+self._minibatch_size

			last_minibatch_of_epoch_is_reached = (trgt > self._no_of_samples)

			if last_minibatch_of_epoch_is_reached:
				trgt = self._no_of_samples
			else:
				self._current_batch_idx += self._minibatch_size

			minibatch_idxs = self._dataset_order_indices[src:trgt]
			minibatch_idxs = np.sort(minibatch_idxs) # required for h5py access
			minibatch_datatensor = self._data_tensor[minibatch_idxs, :,:,:]
			minibatch_labels		 = self._label_list[minibatch_idxs, :]
			
			if last_minibatch_of_epoch_is_reached:
				self._current_batch_idx = 0
				np.random.shuffle(self._dataset_order_indices)

		elif sampling_policy == 'loss_priority':
			print('[AutoBatcher] Not implemented yet.')
		elif sampling_policy == 'mixup':
			print('[AutoBatcher] Not implemented yet.')
				
		return last_minibatch_of_epoch_is_reached, minibatch_datatensor, minibatch_labels


	def getCategoryCount(self):
		return self._no_of_classes


	@staticmethod
	def _inMemoryTrainingIsPossible(memory_fraction, fp_num_count):
		if memory_fraction > 1.0:
			print('[AutoBatcher] Memory fraction cannot be larger than 1.0. Setting to default value of 1.0')
			memory_fraction = 1.0
		available_mem_kb = 0
		with open('/proc/meminfo') as f:
			while True:
				line = f.readline()
				if "MemAvailable:" in line:
					available_mem_kb = int(''.join([s for s in line if s.isdigit()]))
					break
		
		required_mem_kb = int(4*(fp_num_count)/1000) # each float32 takes 32 bit == 4 byte of storage
		available_mem_kb = int(available_mem_kb*memory_fraction)

		# both are now kilobytes
		print( "Memory Requirement of Dataset: {} kB - Allowed Memory: {} kB".format(required_mem_kb, available_mem_kb) )
		return available_mem_kb > required_mem_kb
