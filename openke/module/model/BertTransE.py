import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from .Model import Model

from openke.data.BertTrainDataLoader import bert_model_name, get_id2desToken


# 定义一个函数，用于将张量分割成指定大小的小块
def chunks(tensor, chunk_size):
    for i in range(0, tensor.size(0), chunk_size):
        yield tensor[i:i + chunk_size]


class BertTransE(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None,
                 max_num_tokens=50):
        super(BertTransE, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        self.bert_model_name = bert_model_name
        self.bert_config = AutoConfig.from_pretrained(self.bert_model_name)
        self.bert_config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(self.bert_model_name, config=self.bert_config)

        embedding_dim = self.bert_config.hidden_size

        self.fc = nn.Linear(embedding_dim, dim)
        self.tanh = torch.nn.Tanh()
        self.max_num_tokens = max_num_tokens

        self.pooling = 'mean'

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False
        self.bertEmbedding = None

        # self.scoreWeights = nn.Parameter(torch.Tensor([[1.0] * 4] * self.rel_tot), requires_grad=True)

    def _calc(self, h, t, r, hd, td, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
            hd = F.normalize(hd, 2, -1)
            td = F.normalize(td, 2, -1)
            # hsd=torch.cat((h,hd),dim=-1)
            # tsd=torch.cat((t,td),dim=-1)
            # hsd = F.normalize(hsd, 2, -1)
            # tsd = F.normalize(tsd, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
            hd = hd.view(-1, r.shape[0], hd.shape[-1])
            td = td.view(-1, r.shape[0], td.shape[-1])
            # hsd=torch.cat((h,hd),dim=-1)
            # tsd=torch.cat((t,td),dim=-1)
        if mode == 'head_batch':
            score1 = h + (r - t)
            score2 = hd + (r - td)
            # score3 = h + (r - td)
            # score4 = hd + (r - t)
        else:
            score1 = (h + r) - t
            score2 = (hd + r) - td
            # score3 = (h + r) - td
            # score4 = (hd + r) - t
        score = [torch.norm(score1, self.p_norm, -1).flatten() ,
                 torch.norm(score2,self.p_norm,-1).flatten(),
                 ]
        # torch.norm(score3, self.p_norm, -1).flatten(),
        # torch.norm(score4, self.p_norm, -1).flatten()
        return score

    def forward(self, data, model="train"):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        if model == "train":
            hd = self._encode(self.bert, data['desDictList']['head']['entity_token_ids'],
                              data['desDictList']['head']['entity_mask'],
                              data['desDictList']['head']['entity_token_type_ids'])
            td = self._encode(self.bert, data['desDictList']['tail']['entity_token_ids'],
                              data['desDictList']['tail']['entity_mask'],
                              data['desDictList']['tail']['entity_token_type_ids'])
        else:
            with torch.no_grad():

                if self.bertEmbedding == None:
                    # 定义编码后的张量列表
                    encoded_tensors = []
                    chunk_size = 4000
                    # 分批次对 'head' 进行编码
                    local_id2desToken = get_id2desToken()
                    head_entity_token_ids = local_id2desToken['entity_token_ids'].cuda()
                    head_entity_mask = local_id2desToken['entity_mask'].cuda()
                    head_entity_token_type_ids = local_id2desToken['entity_token_type_ids'].cuda()

                    for chunk_ids, chunk_mask, chunk_token_type_ids in zip(chunks(head_entity_token_ids, chunk_size),
                                                                           chunks(head_entity_mask, chunk_size),
                                                                           chunks(head_entity_token_type_ids,
                                                                                  chunk_size)):
                        hd_chunk = self._encode(self.bert, chunk_ids, chunk_mask, chunk_token_type_ids)
                        encoded_tensors.append(hd_chunk)
                    # 将所有编码后的小张量拼接成一个大张量
                    self.bertEmbedding = torch.cat(encoded_tensors, dim=0)
                if (batch_h.shape[0] == self.ent_tot):
                    hd = self.bertEmbedding
                    td = hd[batch_t]
                else:
                    td = self.bertEmbedding
                    hd = td[batch_h]

        score = self._calc(h, t, r, hd, td, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        scores = self.forward(data, model="predict")
        score=sum(scores)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        hidden_states = outputs['hidden_states']
        # last_hidden_state = outputs.last_hidden_state
        hidden_state = hidden_states[-1]
        cls_output = hidden_state[:, 0, :]
        cls_output = _pool_output(self.pooling, cls_output, mask, hidden_state)
        cls_output = self.fc(cls_output)
        cls_output = self.tanh(cls_output)
        return cls_output


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':  # 对最后一层的所有token进行平均
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
