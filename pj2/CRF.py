import sklearn_crfsuite
import pickle
# 将观测序列中的单词转化为特征字典
def word2features(sentence, i):
    word = sentence[i]
    features = {
        'bias': 1.0,  # 偏置常数
        'word': word,  # 当前字符(单词\汉字)
        'word.isdigit()': word.isdigit(),  # 当前字符是否为数字
        'word.is_word()': is_word(word),   # 当前字符是否为单词
    }

    # 如果不是序列的第一个字符
    if i > 0:
        word = sentence[i - 1]
        # 添加关于上一个字符的特征
        features.update({
            '-1:word': word,
            '-1:word.isdigit()': word.isdigit(),
            '-1:word.is_word()': is_word(word),
        })
    else:
        # 若该字符为序列开头，则增加特征首字符特征
        features['BOS'] = True

    # 如果不是序列的最后一个字符
    if i < len(sentence) - 1:
        word = sentence[i + 1]
        # 添加关于下一个观测值的特征
        features.update({
            '+1:word': word,
            '+1:word.isdigit()': word.isdigit(),
            '+1:word.is_word()': is_word(word),
        })
    else:
        # 若该字符为序列结尾，则增加尾字符特征
        features['EOS'] = True
    return features

#  判断字符是否为单词
def is_word(s):
    if len(s) > 1:
        return True
    if 97 <= ord(s.lower()) <= 122:
        return True
    return False


def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        inputs = []
        sentence = []
        tags = []
        for line in data:
            if line == '\n':
                inputs.append([sentence, tags])
                sentence = []
                tags = []
                continue
            word, tag = line.split(' ')
            tag = tag.strip()
            sentence.append(word)
            tags.append(tag)
        inputs.append([sentence, tags])
    return inputs


def sentence2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]

def write_result(test_data, predicts, path):
    x_test = [t[0] for t in test_data]
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(len(x_test)):
            sentence = x_test[i]
            for j in range(len(sentence)):
                word = sentence[j]
                f.write(word)
                f.write(' ')
                f.write(predicts[i][j])
                f.write('\n')
            if i < len(x_test)-1:
                f.write('\n')

# 数据读取
train_data = load_data('./Chinese/train.txt')
test_data = load_data('./Chinese/validation.txt')
x_train = [sentence2features(t[0]) for t in train_data]
y_train = [t[1] for t in train_data]
x_test = [sentence2features(t[0]) for t in test_data]
# crf模型初始化,采用拟牛顿法进行参数学习
model = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=80, all_possible_transitions=True, c1=0.01, c2=0.01)
# 训练
model.fit(x_train, y_train)
with open('./model/C_CRF.pkl','wb') as f:
    pickle.dump(model,f)
# 预测
# predicts = model.predict(x_test)
# write_result(test_data, predicts)