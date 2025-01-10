import re
import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score,accuracy_score

# 定义标签类别
ABSTAIN = -1
正向 = 0
中性 = 1
负向 = 2


# 优化后的标注函数
@labeling_function()
def lf_正向(x):
    keywords = [
        "如何操作","如何购买","很感兴趣","听起来不错","值得考虑","正合我意","挺有道理","很有帮助","感谢建议","感谢讲解","认可这个方案","认同你的观点"
        ,"觉得可行","进一步了解","约好时间","准备试试","愿意尝试","确实有用","收获很大","满意","靠谱","符合需求","很好的建议","很好的思路","非常赞同",
        "很实用的方法","很专业的讲解","已经了解了","有了方向","会去关注","打算按此操作","约定时间"
    ]
    pattern = re.compile("|".join(keywords))
    return 正向 if pattern.search(x.text) else ABSTAIN

@labeling_function()
def lf_中性(x):
    keywords = [
        "先了解","再琢磨","需要时间","缓缓再说","过后想想","之后再议",
        "暂时不定","考虑一下","回头再说","等会研究","之后考虑","之后权衡","暂不决定","等以后","下次再谈","以后再说","有空再说","还得想想","再考察"
    ]
    pattern = re.compile("|".join(keywords))
    return 中性 if pattern.search(x.text) else ABSTAIN

@labeling_function()
def lf_负向(x):
    keywords = [
      "不感兴趣","没有必要","不考虑","不需要","不满意","不信任","不想做决定","感到失望","失望","拒绝",
      "不做","没兴趣","不需要","没信心","怀疑","不信任","不考虑","不适合","反感","抵触","糟糕","差劲",
      "没用","毫无价值","不认可","不赞同","不接受","不想尝试","不想接触","没意义","不屑","难以接受","不靠谱"
    ]
    pattern = re.compile("|".join(keywords))
    return 负向 if pattern.search(x.text) else ABSTAIN

def generate_weak_labels(data):
    # 应用标注函数
    lfs = [lf_正向, lf_中性, lf_负向]
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=data)
    
    # 分析标注函数
    LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    # 训练生成模型
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # 生成最终标签
    preds_train = label_model.predict(L=L_train)
    data['label'] = preds_train

    return data


def train_and_evaluate_model(data):
    # 分离特征和标签
    X = data['text']
    y = data['label']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义管道
    pipeline = make_pipeline(TfidfVectorizer(), StandardScaler(with_mean=False), SVC())

    # 定义参数网格
    param_grid = {
        'tfidfvectorizer__max_df': [0.8, 0.9, 1.0],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf']
    }

    # 网格搜索
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # 最佳模型
    best_model = grid_search.best_estimator_

    # 交叉验证评分
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f'Cross-validation scores: {cv_scores}')
    print(f'Mean cross-validation score: {cv_scores.mean()}')

    # 预测
    y_pred = best_model.predict(X_test)

    # 输出分类报告
    print(classification_report(y_test, y_pred))

    return best_model

def predict_from_csv(file_path, model):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 预测
    predictions = model.predict(data['text'])

    # 将预测结果添加到数据框
    data['predicted_label'] = predictions

    return data


def predict_from_csv(file_path, model):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 预测
    predictions = model.predict(data['text'])

    # 将预测结果添加到数据框
    data['predicted_label'] = predictions

    return data

# 调用示例
# 从CSV文件中读取对话内容
file_path = '/data/text_emotion_classifier/data/emotion/1060_full_emotion.csv'
read_data = pd.read_csv(file_path)

context_data=context_data=pd.DataFrame(read_data["text"])

# 生成弱标签
weak_label_data = generate_weak_labels(context_data)
weak_label_data= weak_label_data[weak_label_data['label'] >= 0]


# 训练和评估模型
model = train_and_evaluate_model(weak_label_data)

# 预测CSV文件中的每行
predicted_data = predict_from_csv(file_path, model)

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
accuracy = accuracy_score(predicted_data ['labels'],predicted_data['predicted_label'])