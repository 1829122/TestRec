import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, seqs): # TODO: fp64 and int64 as default in python, trim?
        seqs = self.emb_dropout(seqs)
        #print(seqs.shape)
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs): # for training 
        #print(log_seqs)       
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

       


        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return log_feats



class FFN(torch.nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()
        self.n_head = args.n_head
        self.hidden_units = args.hidden_units
        self.dropout_rate = args.dropout_rate
        self.dev = args.device
        self.Relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(self.hidden_units // self.n_head, self.hidden_units // self.n_head)
        self.fc2 = torch.nn.Linear(self.hidden_units // self.n_head, self.hidden_units // self.n_head)
        self.drop_out = torch.nn.Dropout(p = self.dropout_rate)
        self.norm = torch.nn.LayerNorm(self.hidden_units // self.n_head, eps=1e-8)
        self.gate = torch.nn.Linear(self.hidden_units, self.n_head, bias = False)
        self.fc = self.fc.to(self.dev)
        self.fc2 = self.fc2.to(self.dev)
        self.norm = self.norm.to(self.dev)
        self.gate = self.gate.to(self.dev)
    def forward(self, x):
        batch_size, seq_length, hidden = x.size()
        x = x.to(self.dev)
        split_x = torch.split(x, self.hidden_units // self.n_head, dim = -1)
        split_x = [chunk.to(self.dev) for chunk in split_x]
        output = []
        for i in range(self.n_head):
            padding_mask = (split_x[i] == 0)
            head_x = self.fc(split_x[i])
            head_x[padding_mask] = 0
            head_x = self.Relu(head_x)
            head_x = self.fc2(head_x)
            head_x[padding_mask] = 0
            head_x = self.drop_out(head_x)
            head_x = self.norm(head_x + split_x[i])
            output.append(head_x)
        output = torch.cat(output, dim = -1) # (bactch, seq, emb)
        #print(output.shape)
        gate_input = output
        gate_output = self.gate(gate_input)
        output = output.view(batch_size, seq_length, self.n_head, -1)
        gate_output = torch.nn.functional.softmax(gate_output, dim = -1)
        gate_output = gate_output.unsqueeze(-1)
        output = gate_output * output
        output = output.view(batch_size, seq_length, -1)
        return output



class MoEMultiHead(torch.nn.Module):
    def __init__(self, maxlen, hidden_len, n_head, n_expert, dropout_rate, device, args):
        super(MoEMultiHead, self).__init__()
         
        self.maxlen = maxlen # (3,4,12)
        self.args = args
        self.head = n_head # 3
        self.dropout_rate = dropout_rate
        self.expert = n_expert # 3
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.Layernorm = torch.nn.LayerNorm(hidden_len // self.head, eps=1e-8)
        if hidden_len % self.head != 0:
            raise ValueError(f"Not size") 
        self.head_unit = hidden_len // self.head # 12 // 3 = 4
        self.w_q = torch.nn.ModuleList([torch.nn.Linear(self.head_unit, self.head_unit, bias = False) for _ in range(self.expert * self.head)])
        self.w_k = torch.nn.ModuleList([torch.nn.Linear(self.head_unit, self.head_unit, bias = False) for _ in range(self.head)])
        self.w_v = torch.nn.ModuleList([torch.nn.Linear(self.head_unit, self.head_unit, bias = False) for _ in range(self.head)])
        self.attention_mask = ~torch.tril(torch.ones((self.maxlen, self.maxlen), dtype=torch.bool))
        self.attention_mask = self.attention_mask.float().masked_fill(self.attention_mask == 1, float('-inf'))
        self.gate = torch.nn.ModuleList([torch.nn.Linear(self.head_unit * self.expert, self.expert) for _ in range(self.head)])
        self.head_layer = torch.nn.ModuleList([torch.nn.Linear(self.head_unit * self.head, self.head_unit) for _ in range(self.head)])
        self.head_layer.to(device)
        self.attention_mask = self.attention_mask.to(device)  # Move to device
    def forward(self, x):
        batch_size, seq_len, hidden_len = x.size() # 3 4 12
        
        
        model = FFN(self.args)
        f_n = []
        cnt = 0
        for i in range(self.head):
            split_x = self.head_layer[i](x)
            output = []
            for j in range(self.expert):
                padding_mask = (split_x == 0)  # True for padding indices
                #print(cnt + j)
                q = self.w_q[cnt + j](split_x)
                #q[padding_mask] = 0
                k = self.w_k[i](split_x)
                #k[padding_mask] = 0
                #print(split_x[i])
                v = self.w_v[i](split_x)
                #v[padding_mask] = 0
                qk = torch.matmul(q, k.transpose(-2, -1))
                qk *= (self.head_unit ** 0.5)
                qk += self.attention_mask
                #print(v)
                score = torch.nn.functional.softmax(qk, dim = -1)
                
                qkv = torch.matmul(score, v)
                qkv[padding_mask] = 0
                #print(qkv)
                output.append(qkv)
            cnt = cnt + self.head
            
            output = torch.cat(output, dim = -1)
            gate_score = self.gate[i](output)
            gate_score = torch.nn.functional.softmax(gate_score, dim = -1)

            output = output.view(batch_size, seq_len, self.expert, -1)
            output = output * gate_score.unsqueeze(-1)
            output = output.sum(dim=2)
            output = self.dropout1(output)
            output = self.Layernorm(output + split_x[i])
            f_n.append(output)

        f_n = torch.cat(f_n, dim = -1)
        ff = model(f_n)
        return ff    

class RRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(RRec, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.head = args.n_head
        self.expert = args.n_expert
        self.batch_size = args.batch_size
        self.hidden_units = args.hidden_units
        self.max_len = args.maxlen
        self.n_neg = args.n_neg
        self.all_item = torch.arange(self.item_num + 1)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.model = MoEMultiHead(args.maxlen, args.hidden_units, args.n_head, args.n_expert, args.dropout_rate, self.dev, args)
        self.lable_layer = torch.nn.ModuleList([torch.nn.Linear(args.hidden_units, args.hidden_units // args.n_head, bias = False) for _ in range(args.n_head)])
        self.model2 = torch.nn.ModuleList([
            SASRec(user_num, item_num, args) for _ in range(args.num_blocks)  # n_layers 만큼 SASRec 생성
            ])
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Embedding 이전 입력 데이터 출력
        #print("Input log_seqs:", log_seqs)
       
        pos_head = []
        neg_head = []
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))  # Embedding 통과
        
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # 2 4 4
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # B, N, Neg, E
        #print(neg_embs.shape)
        #pos_embs = pos_embs.view(batch, seq, self.head, -1) # 2 4 2 2
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        # Embedding 후 데이터 출력
        #print("After Embedding log_seqs:", log_seqs)
        for layer in self.model2:
            seqs = layer(seqs)
       
        log_feats = self.model(seqs)  # MoEMultiHead 통과 (batch, seq, head, head_emb)
        all_item_emb = self.item_emb(torch.LongTensor(self.all_item).to(self.dev))
        #neg_embs = neg_embs.view(-1, neg_embs.shape[2], neg_embs.shape[3])  # [B*N, Neg, E]
        for i in range(self.head):  # Head 개수만큼 반복
            all_item = self.lable_layer[i](all_item_emb)
            pos_head.append(all_item) # 2 4 2 2 4 2 
            

    # Head 결합
        pos_head = torch.cat(pos_head, dim=-1)  # [B, N, H]
        #neg_head = torch.cat(neg_head, dim=-1)  # [B*N, Neg, H] 8 100 4

    # Negative Head를 원래 배치로 복원
        #neg_head = neg_head.view(self.batch_size, self.max_len,self.n_neg, neg_head.shape[-1])  # [B, N, Neg, H]

    # 정답 및 오답 스코어 계산
        score = torch.matmul(log_feats, pos_head.T)
        #pos_score = (log_feats * pos_head).sum(dim=-1)  # [B, N]
        #neg_score = (log_feats.unsqueeze(2) * neg_head).sum(dim=-1)  # [B, N, Neg]

        return log_feats, score
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        
        for layer in self.model2:
            seqs = layer(seqs)
       
        log_feats = self.model(seqs)  # MoEMultiHead 통과 (batch, seq, head, head_emb)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        pos_head = []
        for i in range(self.head):
            pos = self.lable_layer[i](item_embs)
            pos_head.append(pos)

        pos_head = torch.cat(pos_head, dim = -1)
        logits = pos_head.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)



