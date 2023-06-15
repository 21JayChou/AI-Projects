import random
import numpy as np
import pickle


def load_data(path):
    w_dic = {}
    t_dic = {}
    td = {}
    index2tag = []
    # 用于处理未知词
    w_dic[' '] = 1
    i = 1
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
            if word not in td:
                td[word] = 1
            else:
                td[word] += 1
            if tag not in t_dic:
                t_dic[tag] = j
                j += 1
                index2tag.append(tag)
            sentence.append(word)
            tags.append(tag)
        train_data.append([sentence, tags])
    return w_dic, t_dic, train_data, index2tag


def load_testdata(path):
    with open(path, 'r',encoding='utf-8') as f:
        data = f.readlines()
        test_data = []
        sentence = []
        tags = []
        for line in data:
            if line == '\n':
                test_data.append([sentence, tags])
                sentence = []
                tags = []
                continue
            word, tag = line.split(' ')
            tag = tag.strip()
            sentence.append(word)
            tags.append(tag)
        test_data.append([sentence, tags])
    return test_data

class HMM:
    def __init__(self, w_dic, t_dic, index2tag):
        self.w_dic = w_dic
        self.t_dic = t_dic
        self.index2tag = index2tag
        self.transform = np.zeros([len(t_dic), len(t_dic)])
        self.emit = np.zeros([len(t_dic), len(w_dic)])
        self.initial = np.zeros(len(t_dic))

    def train(self, train_data, smooth=1e-5):
        for sentence, tags in train_data:
            for i in range(len(sentence)):
                word = sentence[i]
                tag = tags[i]
                if i == 0:
                    self.initial[self.t_dic[tag]] += 1
                if i > 0:
                    pre_tag = tags[i-1]
                    # 依据当前状态和前一状态，使状态转移矩阵对应位置频数加1
                    self.transform[self.t_dic[pre_tag]][self.t_dic[tag]] += 1
                # 根据当前状态和观测值，使观测概率矩阵对应位置频数加1
                self.emit[self.t_dic[tag]][self.w_dic[word]] += 1
        # 依据频数算出频率作为概率，加上smooth防止出现概率为0的情况
        self.initial = self.initial/(len(train_data))
        self.transform += smooth
        self.transform = self.transform/np.sum(self.transform, axis=1, keepdims=True)
        self.emit += smooth
        self.emit = self.emit/np.sum(self.emit, axis=1, keepdims=True)

    def Viterbi(self, test_data, result_path):
        predicts = []
        for i in range(len(test_data)):
            sentence = test_data[i][0]
            # maxp计算在给定观测序列下的所有状态序列中的最大概率
            # path保存在每一个时刻不同状态对应的最大概率的状态序列的前一个状态，用于回溯解码
            maxP = np.zeros([len(sentence),len(self.t_dic)])
            path = np.zeros([len(sentence)-1,len(self.t_dic)])
            for j in range(len(sentence)):
                # O为句子中的某个字或单词对应在观测概率矩阵中的列下标
                if sentence[j] not in self.w_dic:
                    O = self.w_dic[' ']
                else:
                    O = self.w_dic[sentence[j]]
                if j == 0:
                    maxP[0] = self.initial*self.emit[:, O]
                else:
                    t = maxP[j-1]*self.transform.T
                    maxP[j] = np.amax(t, axis=1)*self.emit[:, O]
                    path[j-1] = np.argmax(t, axis=1)
            result = []
            index = np.argmax(maxP[len(sentence)-1])
            result.append(self.index2tag[index])
            for j in range(len(sentence)-2, -1, -1):
                index = int(path[j][index])
                result.append(self.index2tag[index])
            result.reverse()
            predicts.append(result)
            #  将预测结果写入文件
        with open(result_path,'w',encoding='utf-8') as f:
            for i in range(len(test_data)):
                sentence = test_data[i][0]
                for j in range(len(sentence)):
                    word = sentence[j]
                    f.write(word)
                    f.write(' ')
                    f.write(predicts[i][j])
                    f.write('\n')
                f.write('\n')


w_dic, t_dic, train_data, index2tag = load_data('./Chinese/train.txt')
model = HMM(w_dic, t_dic, index2tag)
model.train(train_data)
with open('./model/C_HMM.pkl','wb+') as f:
    pickle.dump(model,f)

