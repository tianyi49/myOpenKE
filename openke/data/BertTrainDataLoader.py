# coding:utf-8
import os
import ctypes
import json
from typing import Optional, List
import torch
import time

import numpy as np
from transformers import AutoTokenizer

bert_model_name = 'bert_uncased_L-2_H-128_A-2'

tokenizer: AutoTokenizer = None
def get_tokenizer():
    if tokenizer is None:
        build_tokenizer()
    return tokenizer
def build_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)


class TrainDataSampler(object):
	def __init__(self, nbatches, datasampler):
		self.nbatches = nbatches
		self.datasampler = datasampler
		self.batch = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.batch += 1
		if self.batch > self.nbatches:
			raise StopIteration()
		return self.datasampler()

	def __len__(self):
		return self.nbatches

class TrainDataLoader(object):

	def __init__(self,
		in_path = "./",
		tri_file = None,
		ent_file = None,
		rel_file = None,
		batch_size = None,
		nbatches = None,
		threads = 8,
		sampling_mode = "normal",
		bern_flag = False,
		filter_flag = True,
		neg_ent = 1,
		neg_rel = 0):

		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		"""argtypes"""
		self.lib.sampling.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64
		]
		self.in_path = in_path
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"
			self.des_file = in_path + "entityDesDict.json"
		"""set essential parameters"""
		self.work_threads = threads
		self.nbatches = nbatches
		self.batch_size = batch_size
		self.bern = bern_flag
		self.filter = filter_flag
		self.negative_ent = neg_ent
		self.negative_rel = neg_rel
		self.sampling_mode = sampling_mode
		self.cross_sampling_flag = 0

		self.read()

	def read(self):
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		else:
			self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
			self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
			self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))

		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.work_threads)
		self.lib.randReset()
		self.lib.importTrainFiles()
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.tripleTotal = self.lib.getTrainTotal()

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
		self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
		self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
		self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

	def sampling(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t,
			"batch_r": self.batch_r,
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	def sampling_head(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	def sampling_tail(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""interfaces to set essential parameters"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""interfaces to get essential parameters"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches

id2desToken=None
def get_id2desToken():
	return id2desToken
def set_id2desToken(val):
	global id2desToken
	id2desToken=val


def get_id2desToken():
	return id2desToken

class BertTrainDataLoader(object):

	def __init__(self,
				 in_path="./",
				 tri_file=None,
				 ent_file=None,
				 rel_file=None,
				 batch_size=None,
				 nbatches=None,
				 threads=8,
				 sampling_mode="normal",
				 bern_flag=False,
				 filter_flag=True,
				 neg_ent=1,
				 neg_rel=0,max_num_tokens=50):

		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		"""argtypes"""
		self.lib.sampling.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64,
			ctypes.c_int64
		]
		self.in_path = in_path
		self.tri_file = tri_file
		self.ent_file = ent_file
		self.rel_file = rel_file
		if in_path != None:
			self.tri_file = in_path + "train2id.txt"
			self.ent_file = in_path + "entity2id.txt"
			self.rel_file = in_path + "relation2id.txt"
		self.max_num_tokens =max_num_tokens


		"""set essential parameters"""
		self.work_threads = threads
		self.nbatches = nbatches
		self.batch_size = batch_size
		self.bern = bern_flag
		self.filter = filter_flag
		self.negative_ent = neg_ent
		self.negative_rel = neg_rel
		self.sampling_mode = sampling_mode
		self.cross_sampling_flag = 0
		self.read()
		global id2desToken
		self.id2desToken=id2desToken
		if self.id2desToken is None:
			"""read entity description and id2desToken"""
			des_dict = json.load(open(in_path + "entityDesDict.json"))["id2des"]

			self.id2desToken = self.init_vectorize(des_dict)

			set_id2desToken(self.id2desToken)

	def read(self):
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		else:
			self.lib.setTrainPath(ctypes.create_string_buffer(self.tri_file.encode(), len(self.tri_file) * 2))
			self.lib.setEntPath(ctypes.create_string_buffer(self.ent_file.encode(), len(self.ent_file) * 2))
			self.lib.setRelPath(ctypes.create_string_buffer(self.rel_file.encode(), len(self.rel_file) * 2))

		self.lib.setBern(self.bern)
		self.lib.setWorkThreads(self.work_threads)
		self.lib.randReset()
		self.lib.importTrainFiles()
		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.tripleTotal = self.lib.getTrainTotal()

		if self.batch_size == None:
			self.batch_size = self.tripleTotal // self.nbatches
		if self.nbatches == None:
			self.nbatches = self.tripleTotal // self.batch_size
		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)

		self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
		self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
		self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
		self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
		self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
		self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

	def sampling(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			0,
			self.filter,
			0,
			0
		)

		desDictList={"head":{},"tail":{}}
		desDictList["head"]['entity_token_ids']=self.id2desToken['entity_token_ids'][self.batch_h].cuda()
		desDictList["head"]['entity_mask']=self.id2desToken['entity_mask'][self.batch_h].cuda()
		desDictList["head"]['entity_token_type_ids']=self.id2desToken['entity_token_type_ids'][self.batch_h].cuda()
		desDictList["tail"]['entity_token_ids']=self.id2desToken['entity_token_ids'][self.batch_t].cuda()
		desDictList["tail"]['entity_mask']=self.id2desToken['entity_mask'][self.batch_t].cuda()
		desDictList["tail"]['entity_token_type_ids']=self.id2desToken['entity_token_type_ids'][self.batch_t].cuda()
		return {
			"desDictList":desDictList,
			"batch_h": self.batch_h,
			"batch_t": self.batch_t,
			"batch_r": self.batch_r,
			"batch_y": self.batch_y,
			"mode": "normal"
		}

	def sampling_head(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			-1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h,
			"batch_t": self.batch_t[:self.batch_size],
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "head_batch"
		}

	def sampling_tail(self):
		self.lib.sampling(
			self.batch_h_addr,
			self.batch_t_addr,
			self.batch_r_addr,
			self.batch_y_addr,
			self.batch_size,
			self.negative_ent,
			self.negative_rel,
			1,
			self.filter,
			0,
			0
		)
		return {
			"batch_h": self.batch_h[:self.batch_size],
			"batch_t": self.batch_t,
			"batch_r": self.batch_r[:self.batch_size],
			"batch_y": self.batch_y,
			"mode": "tail_batch"
		}

	def cross_sampling(self):
		self.cross_sampling_flag = 1 - self.cross_sampling_flag
		if self.cross_sampling_flag == 0:
			return self.sampling_head()
		else:
			return self.sampling_tail()

	"""interfaces to set essential parameters"""

	def set_work_threads(self, work_threads):
		self.work_threads = work_threads

	def set_in_path(self, in_path):
		self.in_path = in_path

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
		self.nbatches = self.tripleTotal // self.batch_size

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_bern_flag(self, bern):
		self.bern = bern

	def set_filter_flag(self, filter):
		self.filter = filter

	"""interfaces to get essential parameters"""

	def get_batch_size(self):
		return self.batch_size

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.tripleTotal

	def _custom_tokenize(self, text: str,
						 text_pair: Optional[str] = None) -> dict:
		tokenizer = get_tokenizer()
		encoded_inputs = tokenizer(text=text,
										text_pair=text_pair if text_pair else None,
										add_special_tokens=True,
										max_length=self.max_num_tokens,
										return_token_type_ids=True,
										truncation=True)
		return encoded_inputs
	def _bacth_custom_tokenize(self, texts: List[str]) -> dict:
		tokenizer = get_tokenizer()
		encoded_inputs = tokenizer(texts,
								   add_special_tokens=True,
								   max_length=self.max_num_tokens,
								   truncation=True,
								   padding='max_length',  # 如果需要对不同长度的文本进行填充
								   return_token_type_ids=True,
								   return_attention_mask=True,
								   return_tensors='pt')
		return encoded_inputs
	def vectorize(self, head_desc, tail_desc) -> dict:

		head_encoded_inputs = self._custom_tokenize(text=head_desc)

		tail_encoded_inputs = self._custom_tokenize(text=tail_desc)

		return {
			'tail_token_ids': tail_encoded_inputs['input_ids'],
			'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
			'head_token_ids': head_encoded_inputs['input_ids'],
			'head_token_type_ids': head_encoded_inputs['token_type_ids'],
		}
	def init_vectorize(self, entity_desc) -> dict:

		entity_encoded_inputs = self._bacth_custom_tokenize(texts=entity_desc)

		return {
			'entity_token_ids': entity_encoded_inputs['input_ids'],
			'entity_token_type_ids': entity_encoded_inputs['token_type_ids'],
			'entity_mask': entity_encoded_inputs['attention_mask']
		}
	def __iter__(self):
		if self.sampling_mode == "normal":
			return TrainDataSampler(self.nbatches, self.sampling)
		else:
			return TrainDataSampler(self.nbatches, self.cross_sampling)

	def __len__(self):
		return self.nbatches

def to_indices_and_mask(batch_tensor, pad_token_id=0,senlen=50,need_mask=True):
	mx_len = max([t.size(0) for t in batch_tensor])
	batch_size = len(batch_tensor)
	mx_len=senlen
	indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
	# For BERT, mask value of 1 corresponds to a valid position
	if need_mask:
		mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
	for i, t in enumerate(batch_tensor):
		indices[i, :len(t)].copy_(t)
		if need_mask:
			mask[i, :len(t)].fill_(1)
	if need_mask:
		return indices, mask
	else:
		return indices
def collate(batch_data) -> dict:
    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data['desDictList']],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data['desDictList']],
        need_mask=False)
    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data['desDictList']],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data['desDictList']],
        need_mask=False)
    return {
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids
	}

def init_collate(desDict) -> dict:
    entity_token_ids, entity_mask = to_indices_and_mask(
        [torch.LongTensor(ex['entity_token_ids']) for ex in desDict],
        pad_token_id=get_tokenizer().pad_token_id)
    entity_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['entity_token_type_ids']) for ex in desDict],
        need_mask=False)
    return {
        'entity_token_ids': entity_token_ids,
        'entity_mask': entity_mask,
        'entity_token_type_ids': entity_token_type_ids
	}