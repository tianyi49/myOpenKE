from typing import List

from .Strategy import Strategy

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		scores = self.model(data)
		loss_res=0
		if type(scores) == list:
			for score in scores:
				p_score = self._get_positive_score(score)
				n_score = self._get_negative_score(score)
				loss_res += self.loss(p_score, n_score)#调用损失函数对象的forward方法，计算和多个负样本的平均损失值
		else:
			p_score = self._get_positive_score(scores)
			n_score = self._get_negative_score(scores)
			loss_res += self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res



class AttNegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(AttNegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		scores,attWeight= self.model(data,self.batch_size)
		loss_res=0
		if type(scores) == list:
			for count,score in enumerate(scores):
				p_score = self._get_positive_score(score)
				n_score = self._get_negative_score(score)
				if count==0:
					loss_res += self.loss(p_score, n_score,(1-attWeight))#调用损失函数对象的forward方法，计算和多个负样本的平均损失值
				else:
					loss_res+=self.loss(p_score, n_score,attWeight)
		else:
			p_score = self._get_positive_score(scores)
			n_score = self._get_negative_score(scores)
			loss_res += self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res