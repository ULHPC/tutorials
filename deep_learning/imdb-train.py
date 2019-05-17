
import tensorflow as tf
from tensorflow import keras
import numpy as np

def prepare_data(vocab_size):
	imdb = keras.datasets.imdb
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
	print("Training entries: {}, labels: {}.".format(len(train_data), len(train_labels)))

	# A dictionary mapping words to an integer index
	word_index = imdb.get_word_index()
	# The first indices are reserved
	word_index = {k:(v+3) for k,v in word_index.items()} 
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2  # unknown
	word_index["<UNUSED>"] = 3
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	def decode_review(text): return ' '.join([reverse_word_index.get(i, '?') for i in text])
	print(decode_review(train_data[12]), train_labels[12])

	train_data = keras.preprocessing.sequence.pad_sequences(
			train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
	test_data = keras.preprocessing.sequence.pad_sequences(
			test_data, value=word_index["<PAD>"], padding='post', maxlen=256)
	return (train_data, train_labels), (test_data, test_labels)

def mtrain(vocab_size, train_data, train_labels):
	# input shape is the vocabulary count used for the movie reviews (10,000 words)
	model = keras.Sequential()
	model.add(keras.layers.Embedding(vocab_size, 16))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(16, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

	#model.summary()

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

	x_val = train_data[:10000]
	partial_x_train = train_data[10000:]
	y_val = train_labels[:10000]
	partial_y_train = train_labels[10000:]

	history = model.fit(partial_x_train, partial_y_train, 
			epochs=40, 
			batch_size=512, 
			validation_data=(x_val, y_val), 
			verbose=1)
	return model

def main(vocab_size=20000):
	print("TF {}, TF.Keras {}.".format(tf.__version__, tf.keras.__version__))
	(train_data, train_labels), (test_data, test_labels) = prepare_data(vocab_size)
	model = mtrain(vocab_size, train_data, train_labels)
	results = model.evaluate(test_data, test_labels)
	print(results)

if __name__ == '__main__':
	main()

