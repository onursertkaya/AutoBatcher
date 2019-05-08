#!/usr/bin/python3

import tensorflow as tf
import sys
sys.path.append('/path/to/autobatcher')
from autobatcher import AutoBatcher

tf.set_random_seed(1)

def main():
	ab = AutoBatcher.loadDatasetFromPath('/path/to/main/dataset/path',img_size=320, memory_fraction=1.0)
	#ab = Autobatcher.loadDatasetFromSerializedFile('/path/to/file/train.hdf5')
	ab.setMiniBatchSize(64)

	output_dim = ab.getCategoryCount()

	model = tf.keras.Sequential([
		tf.keras.layers.Conv2D(32,5,2,'same',activation='relu'),
		tf.keras.layers.Conv2D(32,3,1,'same',activation='relu'),
		tf.keras.layers.Conv2D(32,3,2,'same',activation='relu'),
		tf.keras.layers.Conv2D(32,3,1,'same',activation='relu'),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(output_dim,activation='softmax')
	])

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
								loss='categorical_crossentropy',
								metrics=['accuracy'])


	epochs = 100
	for i in range(epochs):
		print('Epoch {} / {}\n'.format(i,epochs))
		while True:
			epoch_complete,x,y = ab.getNextMiniBatch()
			model.train_on_batch(x,y)
			model.evaluate(x,y)
			if epoch_complete:
				break
		print('\n========\n')


if __name__ == '__main__':
	main()
