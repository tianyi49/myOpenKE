# coding:utf-8
import os
import ctypes
from typing import Optional, List

import torch
import numpy as np
import json
from openke.data.BertTrainDataLoader import get_tokenizer, get_id2desToken,set_id2desToken

class TestDataSampler(object):

	def __init__(self, data_total, data_sampler):
		self.data_total = data_total
		self.data_sampler = data_sampler
		self.total = 0

	def __iter__(self):
		return self

	def __next__(self):
		self.total += 1 
		if self.total > self.data_total:
			raise StopIteration()
		return self.data_sampler()

	def __len__(self):
		return self.data_total

class BertTestDataLoader(object):

	def __init__(self, in_path = "./", sampling_mode = 'link', type_constrain = True,max_num_tokens=50):
		base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		"""for link prediction"""
		self.lib.getHeadBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		self.lib.getTailBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""for triple classification"""
		self.lib.getTestBatch.argtypes = [
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
			ctypes.c_void_p,
		]
		"""set essential parameters"""
		self.in_path = in_path
		self.sampling_mode = sampling_mode
		self.type_constrain = type_constrain
		self.max_num_tokens = max_num_tokens
		self.read()
		self.id2desToken = get_id2desToken()
		if self.id2desToken is None:
			"""read entity description and id2desToken"""
			des_dict = json.load(open(in_path + "entityDesDict.json"))["id2des"]
			self.id2desToken = self.init_vectorize(des_dict)
			set_id2desToken(self.id2desToken)



	def read(self):
		self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
		self.lib.randReset()
		self.lib.importTestFiles()

		if self.type_constrain:
			self.lib.importTypeFiles()

		self.relTotal = self.lib.getRelationTotal()
		self.entTotal = self.lib.getEntityTotal()
		self.testTotal = self.lib.getTestTotal()

		self.test_h = np.zeros(self.entTotal, dtype=np.int64)
		self.test_t = np.zeros(self.entTotal, dtype=np.int64)
		self.test_r = np.zeros(self.entTotal, dtype=np.int64)
		self.test_h_addr = self.test_h.__array_interface__["data"][0]
		self.test_t_addr = self.test_t.__array_interface__["data"][0]
		self.test_r_addr = self.test_r.__array_interface__["data"][0]

		self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
		self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
		self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

	def sampling_lp(self):
		res = []
		self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h.copy(), 
			"batch_t": self.test_t[:1].copy(), 
			"batch_r": self.test_r[:1].copy(),
			"mode": "head_batch"
		})
		self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
		res.append({
			"batch_h": self.test_h[:1],
			"batch_t": self.test_t,
			"batch_r": self.test_r[:1],
			"mode": "tail_batch"
		})
		return res

	def sampling_tc(self):
		self.lib.getTestBatch(
			self.test_pos_h_addr,
			self.test_pos_t_addr,
			self.test_pos_r_addr,
			self.test_neg_h_addr,
			self.test_neg_t_addr,
			self.test_neg_r_addr,
		)
		return [ 
			{
				'batch_h': self.test_pos_h,
				'batch_t': self.test_pos_t,
				'batch_r': self.test_pos_r ,
				"mode": "normal"
			}, 
			{
				'batch_h': self.test_neg_h,
				'batch_t': self.test_neg_t,
				'batch_r': self.test_neg_r,
				"mode": "normal"
			}
		]

	"""interfaces to get essential parameters"""

	def get_ent_tot(self):
		return self.entTotal

	def get_rel_tot(self):
		return self.relTotal

	def get_triple_tot(self):
		return self.testTotal

	def set_sampling_mode(self, sampling_mode):
		self.sampling_mode = sampling_mode
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
	def init_vectorize(self, entity_desc) -> dict:

		entity_encoded_inputs = self._bacth_custom_tokenize(texts=entity_desc)

		return {
			'entity_token_ids': entity_encoded_inputs['input_ids'],
			'entity_token_type_ids': entity_encoded_inputs['token_type_ids'],
			'entity_mask': entity_encoded_inputs['attention_mask']
		}
	def __len__(self):
		return self.testTotal

	def __iter__(self):
		if self.sampling_mode == "link":
			self.lib.initTest()
			return TestDataSampler(self.testTotal, self.sampling_lp)
		else:
			self.lib.initTest()
			return TestDataSampler(1, self.sampling_tc)
		
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