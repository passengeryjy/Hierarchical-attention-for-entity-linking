import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedAttention(nn.Module):

    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.att_w = nn.Linear(att_size,1,bias=False)

    def forward(self,input,len_s):
        att = self.att_w(input).squeeze(-1)
        out = self._masked_softmax(att,len_s).unsqueeze(-1)
        return out
        
    
    def _masked_softmax(self,mat,len_s):
        
        #print(len_s.type())
        len_s = len_s.type_as(mat.data)#.long()
        idxes = torch.arange(0,int(len_s[0]),out=mat.data.new(int(len_s[0])).long()).unsqueeze(1)
        mask = (idxes.float()<len_s.unsqueeze(0)).float()

        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(0,True)+0.0001
     
        return exp/sum_exp.expand_as(exp)

class AttentionalBiRNN(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0, RNN_cell=nn.GRU):
        super(AttentionalBiRNN, self).__init__()
        
        self.natt = hid_size*2

        self.rnn = RNN_cell(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
        self.lin = nn.Linear(hid_size*2,self.natt)
        self.att_w = nn.Linear(self.natt,1,bias=False)
        self.emb_att = EmbedAttention(self.natt)

    
    def forward(self, packed_batch):
        
        rnn_sents,_ = self.rnn(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)

        emb_h = F.tanh(self.lin(enc_sents))

        attended = self.emb_att(emb_h,len_s) * enc_sents
        return attended.sum(0,True).squeeze(0)

class SingleLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class HAN(nn.Module):

    def __init__(self, ntoken, nent, num_class=1, emb_size=200, hid_size=50):
        super(HAN, self).__init__()
    
        self.emb_size = emb_size
        self.embed_doc = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.embed_ent = nn.Embedding(nent, emb_size, padding_idx=0)
        self.word = AttentionalBiRNN(emb_size, hid_size)
        self.sent = AttentionalBiRNN(hid_size*2, hid_size)


    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed_doc.weight.data = emb_tensor
    def set_ent_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed_ent.weight.data = emb_tensor
    
    def _reorder_sent(self,sents,sent_order):
        
        sents = F.pad(sents,(0,0,1,0)) #adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0),sent_order.size(1),sents.size(1))
        return revs
 

    def forward(self, batch_doc, sent_order, length_sent, length_doc, men_sent_idx, men_ent_idx, can_ent_idx, can_ent_index):
        '''
        input:
        batch_doc:排序后的文档token嵌入表示，[bs, num_sent, ]
        sent_order: 每篇文档中的句子长度排序
        length_sent: [bs, num_sent]
        length_doc: [bs, len_doc]
        '''
        #HAN架构
        #对分词后的token序列进行编码，[num_sent, num_token, emb_size]
        emb_w = F.dropout(self.embed_doc(batch_doc), training=self.training)
        #对待候选实体的嵌入进行编码
        emb_ent = self.embed_ent(can_ent_idx)
        #对句子进行长度统一
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, length_sent, batch_first=True)
        #句子嵌入，[num_sent, hid_size]
        sent_embs = self.word(packed_sents)
        #[bs, num_sent, emb_size],从这取句子的嵌入
        doc_embs = self._reorder_sent(sent_embs,sent_order)
        each_sent_embs = doc_embs
        packed_doc = torch.nn.utils.rnn.pack_padded_sequence(doc_embs, length_doc, batch_first=True)
        #获得文档嵌入， [bs, emb_size]，从这取文档嵌入
        doc_embs = self.sent(packed_doc)

        #计算每篇文档中待链接实体的句子、文档向量分别和候选实体向量之间的相似度
        '''
        each_sent_embs: [bs, num_sent, emb_size]张量
        doc_embs: [bs, emb_size]张量
        emb_ent: [bs, num_men, num_ent, emb_size]张量
        men_sent_idx: [bs, num_men]列表
        can_ent_index: 对应候选实体索引，[bs, num_men_each_doc, num_ent_each_men]列表 
        '''
        bs, num_men, num_ent, _ = emb_ent.size()
        #初始化相似度矩阵
        sent_ent_sim = torch.zeros(bs, num_men, num_ent).cuda()
        doc_ent_sim = torch.zeros(bs, num_men, num_ent).cuda()

        for i in range(bs):
            for j in range(len(can_ent_index[i])):
                #文档张量
                vec_d = doc_embs[i]
                #取提及所在句子张量
                vec_s = each_sent_embs[i, men_sent_idx[i][j]]
                for k in range(len(can_ent_index[i][j])):
                    vec_e = emb_ent[i, j, k]
                    sent_ent_sim[i, j, k] = torch.cosine_similarity(vec_s, vec_e, dim=0)
                    doc_ent_sim[i, j, k] = torch.cosine_similarity(vec_d, vec_e, dim=0)

        #计算最终得分
        concat_sim = torch.concat([sent_ent_sim, doc_ent_sim], dim=-1)
        mask = (sent_ent_sim != 0).float()
        input_dim = concat_sim.size(-1)
        output_dim = num_ent  # 最终分数的维度
        mlp = SingleLayerMLP(input_dim, output_dim).cuda()

        out = mlp(concat_sim)
        out = out * mask
        return out



