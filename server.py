# -*- coding: utf-8 -*-

"""
############################
CLASS : 'NORMAL', 'Map', 'YOUTUBE', 'WIKI', 'CALENDER', 'shopping', 'email', 'hotel', 'restaurant', 'CES', 'Flight'
####################
"""
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
import getopt
import time
import sys
import os
import requests
import math
from googletrans import Translator

from dnc.dnc import DNC
from recurrent_controller import RecurrentController

from grpc_wrapper.server import create_server, BaseModel
from grpc_wrapper.client import create_client
from my_utils import *


if __name__ == '__main__':

	translator = Translator()
	keywords = ['where', 'video', 'mail', 'schedule', 'address', 'know', 'weather', 'reserv', 'flight', 'shop',
				'restaurant', 'wonder', 'door', 'news', 'movie', 'stock', 'summar', 'depress', 'sport', 'book']
	memories = {}

	dirname = os.path.dirname(__file__)
	# ckpts_dir = os.path.join(dirname, 'checkpoints/')
	pkl_data_file = os.path.join(dirname, 'intention_2019_10_13.pkl')
	ckpt_path = os.path.join(dirname, 'checkpoints/intention/step-40000/model.ckpt')
	pkl_data = pickle.load(open(pkl_data_file, 'rb'))

	# train_data = pkl_data['train']

	# ACT_list = pkl_data['act']
	# ENTITY = pkl_data['entity']

	inv_dictionary = pkl_data['idx2w']
	lexicon_dict = pkl_data['w2idx']
	target_class = len(pkl_data['class'])

	llprint("Loading Data ... ")

	# dncinput = np.load(input_file)

#	 inv_dictionary = idx2w = pkl_data['idx2w']

	llprint("Done!\n")

	### NE_space_size , ACT_space_size
	# NE_space_size = len(lexicon_dict)
	# ACT_space_size = 4

	batch_size = 1
	input_size = len(lexicon_dict)
	output_size = 512  ##autoencoder LSTM hidden unit dimension
	sequence_max_length = 100
	word_space_size = len(lexicon_dict)
	words_count = 256
	word_size = 128
	read_heads = 4

	iterations = 100000
	start_step = 0	##woo

	options, _ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations=', 'start='])

	hidden_size = 512
	mlp_input = output_size
	llprint("Done!\n")

	for opt in options:
		if opt[0] == '--checkpoint':
			from_checkpoint = opt[1]
		elif opt[0] == '--iterations':
			iterations = int(opt[1])
		elif opt[0] == '--start':
			start_step = int(opt[1])

	graph = tf.Graph()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with graph.as_default():
		with tf.Session(graph=graph, config=config) as session:

			llprint("Building Computational Graph ... ")

			ncomputer = DNC(
				RecurrentController,
				input_size,
				output_size,
				sequence_max_length,
				words_count,
				word_size,
				read_heads,
				batch_size,
				# NE_space_size,
				# ACT_space_size
			)

			output, _ = ncomputer.get_outputs()

			dec_target = tf.placeholder(tf.int32)

			target_onehot = tf.one_hot(dec_target, target_class)

			with tf.variable_scope('logit'):
				W_logit = tf.get_variable('W_logit', [output_size, target_class])
				b_logit = tf.get_variable('b_logit', [target_class])
				out_logit = tf.matmul(tf.expand_dims(output[0, -1, :], axis=0), W_logit) + b_logit

			#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_logit, labels=target_onehot))

			#gradients = optimizer.compute_gradients(loss)

			llprint("Done!\n")

			llprint("Initializing Variables ... ")
			"""
			##세션 시작########################################################
			"""

			session.run(tf.global_variables_initializer())
			llprint("Done!\n")
			var_list = tf.trainable_variables()
			saver = tf.train.Saver(var_list=var_list)
			print([v.name for v in tf.trainable_variables()])

			saver.restore(session, ckpt_path)

			last_100_losses = []

			start = 0 if start_step == 0 else start_step + 1
			end = start_step + iterations + 1

			start_time_100 = time.time()
			end_time_100 = None
			avg_100_time = 0.
			avg_counter = 0

			inputsen = []
			predsen = []
			lenth = 100

			overlap_num = 0
			before_out = []
			init_M = [
				np.ones([batch_size, words_count, word_size]) * 1e-6,
				np.zeros([batch_size, words_count]),
				np.zeros([batch_size, words_count]),
				np.zeros([batch_size, words_count, words_count]),
				np.ones([batch_size, words_count]) * 1e-6,
				np.ones([batch_size, words_count, read_heads]) * 1e-6,
				np.ones([batch_size, word_size, read_heads]) * 1e-6,

			]

			def conversate(input_, m_count, memory_S):
				if memory_S == None:
					New_memory = init_M
				else:
					New_memory = memory_S

				if m_count == 10 :
					New_memory = init_M
					m_count =0

				x = input_
				x = x.lower().strip()
				# act_num = input_["act"]
				# x = x.replace("'", " ' ")
				sentence_without_new_line = x
				x = x.split(" ")

				user_num = []
				for tt in x:
					try:
						user_num.append(lexicon_dict[tt])
					except KeyError:
						user_num.append(lexicon_dict['<unk>'])

				user_num = user_num + [lexicon_dict["<go>"]]

				'''
				# get named entity from ner docker container
				NES = ner_request(sentence_without_new_line)
				#NES = ner_request(input_["sentence"])
				#NE = ENTITY[m_count][ner_request(input_["sentence"])]
				#NE = ENTITY[ner_request(input_["sentence"])]				 
				try:
					NE[0] = NES[0]
					NE[1] = NES[1]
					NE[2] = NES[2]
				except:
					NE = [23619, 23619, 23619]
					
				# get dialog act from classifier da final docker container
				#ACT = ACT_list[m_count][da_request(input_["sentence"])]
				#ACT = da_request(input_["sentence"])
				#ACT = 1
				# ACT = int(da_request(input_["sentence"]))
				ACT = int(da_request(sentence_without_new_line))
				'''

				# input_data, seq_len, NE_data, ACT_data = mode_pre2(user_num, word_space_size, NE, ACT)
				input_data, seq_len = mode_pre(user_num, word_space_size)
				
				outputvec, memory = session.run([
					out_logit, ncomputer.check_memory
				], feed_dict={
					#dec_in: np.expand_dims(np.expand_dims(dec_input, axis=1), axis=0),
					ncomputer.input_data: input_data,
					ncomputer.sequence_length: seq_len,
					ncomputer.m_0: New_memory[0],
					ncomputer.m_1: New_memory[1],
					ncomputer.m_2: New_memory[2],
					ncomputer.m_3: New_memory[3],
					ncomputer.m_4: New_memory[4],
					ncomputer.m_5: New_memory[5],
					ncomputer.m_6: New_memory[6],
					# ncomputer.input_NE0: NE_data[0],
					# ncomputer.input_NE1: NE_data[1],
					# ncomputer.input_NE2: NE_data[2],
					# ncomputer.input_ACT: ACT_data
				})
				intent_list = pkl_data['class']
				New_memory = memory
				pred_index = outputvec[0].argmax()
				#pred_sent = ("%s\t%4f" %(intent_list[pred_index],outputvec[0][pred_index]))
				pred_sent = ("%s" %(pred_index))

				return pred_sent.replace('<unk>', ','), m_count+1, New_memory

			class Generator(BaseModel):
				def __init__(self, m_count = 0):
					self.m_count = m_count
					self.before_sentence = ""
					self.memory_S = None
					print("stand by")

				def send(self, input):
					print("INPUT : " + str(input))
					roomState = int(input["roomState"])
					roomId = int(input["roomId"])
					original_sentence = str(input["sentence"])

					if roomState == 0:  # room create
						memories[roomId] = [self.m_count, self.memory_S]
						return {"output": "true"}
					elif roomState == 1:  # room remove
						del memories[roomId]
						return {"output": "true"}
					elif roomState == 2:  # room chat
						# only translate with roomState : 2 with sentence
						translated_sentence = translator.translate(original_sentence, dest='en').text
						print("Translated INPUT : " + str(translated_sentence))
						# find memory from list based on roomId
						# result, self.m_count, self.memory_S = conversate(input, self.m_count, self.memory_S)
						result, memories[roomId][0], memories[roomId][1] = \
							conversate(translated_sentence, memories[roomId][0], memories[roomId][1])
						alt_result = keyword_check(translated_sentence)
						print("OUTPUT : " + str(result) + ", ALT_OUTPUT : " + str(alt_result))
						# return {"output": str(result)}
						return {"output": str(math.floor((int(result) * 0.01) + int(alt_result)))}
					else:
						print("Unknown situation")

			def run():
				# port = int(sys.argv[1])
				port = 50051

				model = Generator()

				server = create_server(model, ip="[::]", port=port)
				server.start()
				try:
					while True:
						time.sleep(60 * 60 * 24)
				except KeyboardInterrupt:
					server.stop(0)

			run()
