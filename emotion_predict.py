###情感意图  正确率58.4%###
#          准确率   召回率
#  正向       55%       80%
#  中性       62.5%     50%
#  负向       0%        0%

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score,accuracy_score
import jieba
import re
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler

train_path = '/data/text_emotion_classifier/data/emotion/1060_train_emotion.csv'
test_path = '/data/text_emotion_classifier/data/emotion/1060_test_emotion.csv'


def load_stopword():
    f_stop = open('/data/text_emotion_classifier/stopwords_full.txt', encoding='utf-8')  # 自己的中文停用词表
    sw = [line.strip() for line in f_stop]  # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    sw.extend(['说话','你好','喂喂','叔叔','经理','女士'])
    f_stop.close()
    return sw

proper_nouns =  ['暂不决定', '不适合', '下次再谈', '不想接触', '感谢讲解', '需要时间', '怀疑', '值得考虑', '不考虑', '不感兴趣',
                '如何购买', '等以后', '不赞同','觉得可行', '之后考虑', '约好时间', '不认可', '等会研究', '靠谱', '暂时不定', 
                '没信心', '抵触', '已经了解了', '不接受', '还得想想', '难以接受','感到失望', '满意', '打算按此操作', '确实有用',
                '先了解', '有了方向', '有空再说', '很好的思路', '再琢磨', '约定时间', '缓缓再说', '正合我意', '考虑一下', '很有帮助',
                '不屑', '符合需求', '之后再议', '差劲', '很感兴趣', '不满意', '很实用的方法', '没用', '认可这个方案', '过后想想',
                '收获很大', '反感', '没意义', '会去关注', '再考察', '不信任', '之后权衡', '不想尝试', '不做', '糟糕', '拒绝', 
                '准备试试', '不靠谱', '不需要','听起来不错', '没兴趣', '不想做决定', '没有必要', '愿意尝试', '非常赞同',
                '进一步了解', '感谢建议', '挺有道理', '毫无价值', '认同你的观点', '回头再说','以后再说', '失望', '如何操作', '很好的建议', '很专业的讲解',        
                "公募基金", "股票型基金", "债券型基金", "混合型基金", "货币市场基金","公募基金产品", "公募基金发售", "流动性高的公募基金", "公募基金投资",
                "公募基金类型", "起投金额较低的公募基金", "公募基金公开发售", "公募基金收益率","公募基金回撤", "公募基金净值", "公募基金定投", "公募基金固收", 
                "场内公募基金","场外公募基金", "公募基金锁定期", "公募基金多元化配置", "公募基金理财规划","T0", "高抛低吸", "机器模型", "智能", "算法",
                "下单策略", "盯盘", "市场动态","预设条件自动委托", "程序化交易", "扭亏为盈", "个股深度套牢", "日内交易工具","底仓收益增强", "中低频交易", 
                "构建指数增强","资管产品", "资管产品名称", "投资范围广", "资产多元化配置", "资管产品收益率","资管产品回撤","资管产品净值", "资管产品历史业绩", 
                "资管产品风险", "资管产品管理费""ETF", "ETF基金", "股票型ETF", "债券型ETF", "货币市场型ETF", "境内ETF","跨境ETF", "ETF指数", "ETF份额", 
                "ETF交易", "ETF申购", "ETF赎回", "ETF投资","ETF市场", "ETF产品", "ETF跟踪指数","ETF成分股", "ETF价格波动", "ETF流动性","ETF净值", "ETF溢价", 
                "ETF折价", "ETF基金经理", "ETF托管银行", "ETF发行机构","行业ETF", "主题ETF", "宽基ETF", "规模型ETF", "成长型ETF", "价值型ETF","ETF套利", 
                "ETF联接基金", "ETF组合投资", "ETF定投", "ETF资产配置", "场内交易ETF","新客权益", "新客权益包", "新客福利", "投顾产品5折优惠券", 
                "新客-智能条件单", "新客-打新助手", "新客-Level-2行情", "1万湘盾积分"]


for noun in proper_nouns:
    jieba.add_word(noun)

stopwords = load_stopword()

def is_not_cyclic_string(s):
    n = len(s)

    # 遍历所有可能的子串长度
    for length in range(1, n // 2 + 1):
        # 如果n不能被length整除，则跳过该长度
        if n % length != 0:
            continue

        # 提取出子串
        sub_str = s[:length]

        # 拼接子串，看是否能还原原字符串
        reconstructed_str = sub_str * (n // length)

        # 如果拼接后的字符串与原字符串相同，则说明是循环字符串
        if reconstructed_str == s:
            return False

    return True

def custom_tokenizer(sentence):
    sentence = sentence.strip()
    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
    sentence = re.sub(remove_chars, "", sentence)

    sentence_seged = jieba.cut(sentence.strip())
    
    outstr = ''
    for word in sentence_seged:
         if word not in stopwords and is_not_cyclic_string(word):
            if word != '/t':
                outstr += word
                outstr += " "
    return outstr


def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].apply(custom_tokenizer)
    return data


train_data = preprocess_data(train_path)
test_data = preprocess_data(test_path)

#重采样均衡文本
ros = RandomOverSampler()
X = train_data['text']
y = train_data['labels']
X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)
X_resampled = X_resampled.flatten()


model=make_pipeline(TfidfVectorizer(max_df=0.1,min_df=0.01,ngram_range=(1,7)), SVC(C=1))

model.fit(X_resampled, y_resampled)

def predict_from_csv(test_data, model):
    # 预测
    predictions = model.predict(test_data['text'])

    # 将预测结果添加到数据框
    test_data['predicted_label'] = predictions

    return test_data

predicted_data = predict_from_csv(test_data, model)

# 初始化LabelBinarizer用于多分类问题
lb = LabelBinarizer()

# 对原始标签进行二值化处理
y_true = lb.fit_transform(predicted_data ['labels'])
y_pred = lb.transform(predicted_data['predicted_label'])

# 计算每个类别的准确率和召回率
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

# 获取所有唯一的原始标签类别
labels = lb.classes_

# 构建结果DataFrame
results_df = pd.DataFrame({'类别': labels, '准确率': precision, '召回率': recall})
results_df

accuracy = accuracy_score(predicted_data ['labels'],predicted_data['predicted_label'])
accuracy