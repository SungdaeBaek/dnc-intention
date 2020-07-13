import sys
import numpy as np
import pickle


def ner_request(input_txt):
	URL = 'http://localhost:50058/NER'
	params = {'sentence' : input_txt}
	response = requests.get(URL, params = params)
	if response.status_code == 200:
		#print("from NER : ")
		#print(response.json()["entity_words"])
		#print("\n")
		return ' '.join(response.json()["entity_words"])
	else:
		return ''
	
def da_request(input_txt):
	client = create_client(ip="localhost", port=50053)	   
	input = {
		"sentence" : input_txt
	}
	output = client.send(input)
	#print("from DA : ")
	#print(output)
	#print("\n")
	return output["output"]

def linear(_input, _in_ch, _out_ch, _name):
	w = tf.get_variable(name='%s_w' % _name, shape=[_in_ch, _out_ch], dtype=tf.float32,
						initializer=tf.contrib.layers.xavier_initializer())
	b = tf.get_variable(name='%s_b' % _name, shape=[_out_ch], dtype=tf.float32,
						initializer=tf.constant_initializer(0.0))

	return tf.nn.bias_add(tf.matmul(_input, w), b)


def lstm(_current_input, _state, _name):
	dim_hidden = int(_state.get_shape()[-1] / 2)
	weight_matrix = tf.get_variable(_name, [dim_hidden + int(_current_input.get_shape()[-1]), 4 * dim_hidden],
									initializer=tf.contrib.layers.xavier_initializer())
	bf = tf.get_variable('lstm_bf_%s' % _name, [1, dim_hidden], initializer=tf.constant_initializer(1.0))
	bi = tf.get_variable('lstm_bi_%s' % _name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))
	bo = tf.get_variable('lstm_bo_%s' % _name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))
	bc = tf.get_variable('lstm_bc_%s' % _name, [1, dim_hidden], initializer=tf.constant_initializer(0.0))

	c, h = tf.split(_state, 2, 1)
	input_matrix = tf.concat([h, _current_input], 1)
	f, i, o, Ct = tf.split(tf.matmul(input_matrix, weight_matrix), 4, 1)
	f = tf.nn.sigmoid(f + bf)
	i = tf.nn.sigmoid(i + bi)
	o = tf.nn.sigmoid(o + bo)
	Ct = tf.nn.tanh(Ct + bc)
	new_c = f * c + i * Ct
	new_h = o * tf.nn.tanh(new_c)
	new_state = tf.concat([new_c, new_h], 1)

	return new_h, new_state


def llprint(message):
	sys.stdout.write(message)
	sys.stdout.flush()


def load(path):
	return pickle.load(open(path, 'rb'))


def onehot(index, size):
	vec = np.zeros(size, dtype=np.float32)
	try:
		vec[index] = 1.0
	except:
		vec[index - 1] = 1.0
	return vec


def DNC_input_pre(input_data, word_space_size):
	input_vec = np.array(input_data, dtype=np.int32)
	seq_len = input_vec.shape[0]
	input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
	return (
		np.reshape(input_vec, (1, -1, word_space_size)),
		seq_len)


def prepare_sample(sample, target_code, word_space_size):
	input_vec = np.array(sample[0]['inputs'], dtype=np.int32)
	seq_len = input_vec.shape[0]
	weights_vec = np.zeros(seq_len, dtype=np.float32)
	target_mask = (input_vec == target_code)
	output_vec = np.expand_dims(sample[0]['outputs'], 1)
	weights_vec[target_mask] = 1.0
	input_vec = np.array([onehot(code, word_space_size) for code in input_vec])

	return (
		np.reshape(input_vec, (1, -1, word_space_size)),
		output_vec,
		seq_len,
		np.reshape(weights_vec, (1, -1, 1))
	)


def inv_dict(dictionary):
	return {v: k for k, v in dictionary.iteritems()}


def mode_pre(input_data, word_space_size):
	print(input_data)
	mode_input =np.array(input_data, dtype=np.int32)
	input_vec = np.array([onehot(code, word_space_size) for code in mode_input])
	seq_len = input_vec.shape[0]

	return (
		np.reshape(input_vec, (1, -1, word_space_size)),
		seq_len)

def mode_pre2(input_data, word_space_size, NE, ACT):
	NE_space = word_space_size ##Named Entity dictionay size
	ACT_space = 4  ##Named Entity dictionay size

	mode_input = np.array(input_data, dtype=np.int32)
	input_vec = np.array([onehot(code, word_space_size) for code in mode_input])
	seq_len = input_vec.shape[0]
	NE_vec0 = np.array([onehot(NE[0], NE_space) for x in range(seq_len)])
	NE_vec1 = np.array([onehot(NE[1], NE_space) for x in range(seq_len)])
	NE_vec2 = np.array([onehot(NE[2], NE_space) for x in range(seq_len)])

	NE_data = [np.reshape(NE_vec0, (1, -1, NE_space)),np.reshape(NE_vec1, (1, -1, NE_space)),np.reshape(NE_vec2, (1, -1, NE_space))]

	ACT_vec = np.array([onehot(ACT, ACT_space) for x in range(seq_len)])

	return (
		np.reshape(input_vec, (1, -1, word_space_size)),
		seq_len,
		NE_data,
		np.reshape(ACT_vec, (1, -1, ACT_space)))


def keyword_check(input_sentence):
	index_check = 0
	for index, value in enumerate(keywords):
		if keywords[index] in input_sentence:
			index_check = index + 1
	return index_check
