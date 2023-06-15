import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle

def load_data(path):
    # 在词字典中添加'NONE'，代替测试集中未在训练集中出现过的词。添加'PAD'为后续数据进行批处理作为填充。
    w_dic = {'NONE':0, 'PAD':1}
    t_dic = {}
    i = 2
    j = 0
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        train_data = []
        sentence = []
        tags = []
        for line in data:
            if line == '\n':
                train_data.append([sentence, tags])
                sentence = []
                tags = []
                continue
            word, tag = line.split(' ')
            tag = tag.strip()
            if word not in w_dic:
                w_dic[word] = i
                i += 1
            if tag not in t_dic:
                t_dic[tag] = j
                j += 1
            sentence.append(word)
            tags.append(tag)
        train_data.append([sentence, tags])
        # 添加START和END标签
        t_dic['START'] = len(t_dic)
        t_dic['END'] = len(t_dic)
        index2tag = {v:k for k, v in t_dic.items()}
        index2word = {v:k for k, v in w_dic.items()}
        # 最后得到词->下标字典，标签->下标字典，处理后的数据，下标->标签字典，下标->标签字典
    return w_dic, t_dic, train_data, index2tag, index2word

def log_sum_exp(vec):

    max_score, _ = torch.max(vec, dim=1)
    max_score_broadcast = max_score.unsqueeze(1).repeat_interleave(vec.shape[1], dim=1)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

class dataSet(Dataset):
    def __init__(self, w_dic, t_dic, data, index2tag, index2word):
        self.w_dic = w_dic
        self.t_dic = t_dic
        self.data = data
        self.index2tag = index2tag
        self.index2word = index2word
        # self.data为中文汉字和英文标签，将其转化为索引形式
        self.pair_indexes = []
        for sentence, tags in self.data:
            index1 = [self.w_dic.get(w, self.w_dic['NONE']) for w in sentence]
            index2 = [self.t_dic[t] for t in tags]
            self.pair_indexes.append([index1, index2])

    def __getitem__(self, item):
        return self.pair_indexes[item]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):

        sentences = [s for s, t in batch]
        tags = [t for s, t in batch]
        sen_len = [len(s) for s in sentences]
        max_len = max(sen_len)
        #  进行填充，保证同一个批次的序列长度相同，便于一个batch内并行计算
        sentences = [s + [self.w_dic['PAD']]*(max_len-len(s)) for s in sentences]
        tags = [t + [self.t_dic['O']]*(max_len-len(t)) for t in tags]

        sentences = torch.tensor(sentences, dtype=torch.long)
        tags = torch.tensor(tags, dtype=torch.long)
        sen_len = torch.tensor(sen_len, dtype=torch.long)

        return sentences, tags, sen_len


class BiLSTM_CRF(nn.Module):
    def __init__(self, dataset, embedding_num, hidden_num, device='cuda'):
        super(BiLSTM_CRF, self).__init__()
        self.device = device
        self.state = 'train'
        self.t_dic = dataset.t_dic
        self.w_dic = dataset.w_dic
        self.index2word = dataset.index2word
        self.index2tag = dataset.index2tag
        self.words_num = len(self.w_dic)
        self.tag_num = len(self.t_dic)
        self.hidden_num = hidden_num
        self.embedding = nn.Embedding(self.words_num, embedding_num).to(self.device)  # 将词转化为词向量
        self.lstm = nn.LSTM(embedding_num, hidden_num//2, batch_first=True, num_layers=1, bidirectional=True).to(self.device)  # 定义lstm层
        self.linear = nn.Linear(hidden_num, self.tag_num, bias=False).to(device)  # 将lstm层的输出映射到tags上
        #  初始化转移矩阵，并且设置其他标签到START标签和END标签到其他标签的得分最小(-10000)
        self.transition = nn.Parameter(torch.randn(self.tag_num, self.tag_num, device=self.device))
        self.transition.data[:, self.t_dic['START']] = -10000
        self.transition.data[self.t_dic['END'], :] = -10000
        self.norm_layer = nn.LayerNorm(self.hidden_num).to(device)
    def _get_lstm_features(self, sentence, sen_len):
        embeds = self.embedding(sentence)
        pack = nn.utils.rnn.pack_padded_sequence(embeds, sen_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(pack)
        unpack_sen, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True )
        sen_out = self.norm_layer(unpack_sen)
        features = self.linear(sen_out)
        return features

    def real_score(self, features, tags,sen_len):
        score = torch.zeros(features.shape[0], device=self.device)
        # 为标签序列添加START标签
        start = torch.tensor([self.t_dic['START']],device=self.device).unsqueeze(0).repeat(features.shape[0], 1)
        tags = torch.cat([start, tags],dim=1)
        for i in range(features.shape[0]):
            score[i] = torch.sum(self.transition[tags[i, :sen_len[i]], tags[i, 1:1+sen_len[i]]])+\
                       torch.sum(features[i, range(sen_len[i]), tags[i][1:1+sen_len[i]]])
            # 加入最后一个标签到END标签的转移得分
            score[i] += self.transition[tags[i][sen_len[i]], self.t_dic['END']]
        return score

    def total_score(self, features, sen_len):
        # 设置初始得分
        init_score = torch.full((self.tag_num,), -10000., device=self.device)
        init_score[self.t_dic['START']] = 0.
        scores = torch.zeros(features.shape[0], features.shape[1]+1, features.shape[2], dtype=torch.float32,
                                  device=self.device)
        scores[:, 0, :] = init_score

        transition = self.transition.unsqueeze(0).repeat(features.shape[0], 1, 1)
        for i in range(features.shape[1]):
            emit_score = features[:, i, :]
            temp = scores[:, i, :].unsqueeze(2).repeat(1, 1, features.shape[2])+transition+\
                      emit_score.unsqueeze(1).repeat(1, features.shape[2], 1)
            copy = scores.clone()
            copy[:, i+1, :] = log_sum_exp(temp)
            scores = copy
        scores = scores[range(features.shape[0]), sen_len, :]
        final_score = scores + self.transition[:, self.t_dic['END']].unsqueeze(0).repeat(features.shape[0],1)
        return log_sum_exp(final_score)

    def get_loss(self, features, tags, sen_len):
        # 所有路径得分
        total_score = self.total_score(features, sen_len)
        # 标签路径得分
        real_score = self.real_score(features, tags, sen_len)
        # 返回 batch 分数的平均值
        return torch.mean(total_score - real_score)

    def viterbi(self, features):
        backindexes = []
        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_scores = torch.full((1, self.tag_num), -10000., device=self.device)
        init_scores[0][self.t_dic['START']] = 0.
        # 记录前一时间步的分数
        forward_scores = init_scores
        # 传入的为单个序列,在每个时间步上遍历
        for feat in features:
            bptrs_t = []  # 记录每一个时间所有标签的最大分数的来源索引
            currentscores = []  # 记录每个时间所有标签的得分

            # 一个标签一个标签去计算处理
            for next_tag in range(self.tag_num):
                # 前一时间步分数 + 转移到第 next_tag 个标签的概率
                next_tag_scores = forward_scores + self.transition[:, next_tag]
                # 得到最大分数所对应的索引,即前一时间步哪个标签过来的分数最高
                best_tag_id = argmax(next_tag_scores)
                # 将该索引添加到路径中
                bptrs_t.append(best_tag_id)
                # 将此分数保存下来
                currentscores.append(next_tag_scores[0][best_tag_id].view(1))
            # 加上当前时间步的发射概率
            forward_scores = (torch.cat(currentscores) + feat).view(1, -1)
            # 将当前时间步所有标签最大分数的来源索引保存
            backindexes.append(bptrs_t)

        # 手动加入转移到结束标签的概率
        terminal_scores = forward_scores + self.transition[:, self.t_dic['END']]
        # 在最终位置得到最高分数所对应的索引
        best_tag_id = argmax(terminal_scores)
        # 最高分数
        path_score = terminal_scores[0][best_tag_id]

        # 回溯，向后遍历得到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backindexes):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签
        start = best_path.pop()
        # 将路径反转并且由索引映射到对应的标签
        best_path.reverse()
        str_best_path = []
        for i in range(len(best_path)):
            str_best_path.append(self.index2tag[best_path[i]])
        return path_score, str_best_path

    def forward(self, sentence, tags, sen_len):
        features = self._get_lstm_features(sentence, sen_len)
        if self.state == 'train':
            loss = self.get_loss(features, tags, sen_len)
            return loss
        else:
            predicts = []
            for i, feature in enumerate(features):
                predicts.append(self.viterbi(feature[:sen_len[i]])[1])
            return predicts

# dataLoader构建


w_dic, t_dic, train_data, index2tag, index2word = load_data('./Chinese/train.txt')
valid_data = load_data('./Chinese/validation.txt')[2]
train_dataset = dataSet(w_dic, t_dic, train_data, index2tag, index2word)
valid_dataset = dataSet(w_dic, t_dic, valid_data, index2tag, index2word)
train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, pin_memory=False, shuffle=False,
                              collate_fn=valid_dataset.collate_fn)

embedding_num = 50
hidden_num = 200
epochs = 30
device = 'cuda:0'
model = BiLSTM_CRF(train_dataset, embedding_num, hidden_num, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
def train():
    for epoch in range(epochs):
        model.train()
        model.state = 'train'
        for sentences, tags, sen_len in train_dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            # sen_len = sen_len.to(device)

            loss = model.forward(sentences, tags, sen_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(model, valid_dataloader, file_path):
    with torch.no_grad():
        model.state = 'eval'
        with open(file_path, 'w', encoding='utf-8')as f:
            for sentences, tags, sen_len in valid_dataloader:

                sentences = sentences.to(device)
                tags = tags.to(device)
                predicts = model.forward(sentences, tags, sen_len)
                for sentence, predict in zip(sentences, predicts):
                    for word, tag in zip(sentence, predict):
                        f.write(model.index2word[int(word)]+' ')
                        f.write(tag+'\n')
                    f.write('\n')

if __name__ =='__main__':
    train()
    with open('./model/C_BiLSTM-CRF.pkl','wb') as f:
        pickle.dump(model, f)
















