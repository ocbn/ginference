import re
import traceback
from statistics import mean
import gensim
import nltk
import sklweka.jvm as jvm
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords, reuters
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn import preprocessing, metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklweka.dataset import to_nominal_labels
from sklweka.preprocessing import WekaTransformer
from weka.classifiers import Evaluation, Classifier
from weka.core.classes import Random
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from gensim.test.utils import datapath
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel


tr_informal = {"gardaş": "kardeş", "kardaş": "kardeş", "saol": "sağol", "tskrler": "teşekkürler",
               "panpa": "kanka", "asgm": "aşkım", "arklar": "arkadaşlar", "foto": "fotoğraf", "beyen": "beğen",
               "kardo": "kardeş", "galp": "kalp", "gı": "kız", "ii": "iyi", "as": "aleyküm selam", "iii": "iyi",
               "ins": "inşallah", "knk": "kanka", "bro": "kardeş", "tsk": "teşekkürler", "tşk": "teşekkürler",
               "zmn": "zaman", "cnm": "canım", "abe": "abi", "abey": "abi", "ağbi": "abi", "kanks": "kanka",
               "bise": "bir şey", "birsey": "bir şey", "birşey": "bir şey", "aylen": "ailen"}
emoticon_string = r"""
                    (?:
                        [<>]?
                        [<x:;=8]                        # eyes
                        [/\-o\*\']?                     # optional nose
                        [3o\*\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth      
                        |
                        [3o\*\)\]\(\[dDpP/\:\}\{@\|\\]  # mouth
                        [/\-o\*\']?                     # optional nose
                        [<x:;=8]                        # eyes
                        [<>]?
                    )"""
emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)

nltk.download('punkt', quiet=True)
nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)
# nltk.download('wordnet')


rs = 1
fold = 5
mdf = 5
min_term_length = 2
np.random.seed(rs)
torch.manual_seed(rs)
partition_membership = True
filter_name = "weka.filters.supervised.attribute.PartitionMembership"
# filter_options = ['-W', 'weka.classifiers.trees.J48']
# filter_options = ['-W', 'weka.classifiers.trees.REPTree']
# filter_options = ['-W', 'weka.classifiers.trees.RandomTree']
# number of iterations (I) default value for RF is 100
filter_options = ['-W', 'weka.classifiers.trees.RandomForest', '--', '-P', '100', '-I', '10']

classifiers = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=rs)),
    ('SVM', SVC(kernel='linear', random_state=rs)),
    ('NB', GaussianNB()),
    ('MNB', MultinomialNB()),
    ('DT', DecisionTreeClassifier(random_state=rs)),
    ('KNN', KNeighborsClassifier()),
    ('RF', RandomForestClassifier(n_estimators=100, random_state=rs))
]


# def seed_everything(seed=1234):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     from tensorflow import set_random_seed
#     set_random_seed(2)


def basic_preprocess(text):
    text = str(text).lower().replace("\n", "").strip().rstrip()  # Apply lowercase conversion
    # and strip head and tail whitespaces
    text = emoticon_re.sub(r'', text)  # strip emoticons
    text = re.sub(r'http\S+', '', text)  # remove urls
    text = re.sub("[^çğıöşüa-zA-Z]", " ", text)  # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuations
    text = re.sub(' +', ' ', text)  # remove multiple whitespaces
    return text


def reduce_duplicate_letters(base):
    base_list = list(base)
    new_str = []
    for i in range(len(base_list) - 1, 0, -1):
        current = base_list[i]
        previous = base_list[i - 1]
        if current != previous:
            new_str.append(current)
    new_str.append(base_list[0])
    new_str.reverse()
    return ''.join(new_str)


def fixed_prefix_stemming(base):
    return base[0:5] if len(base) > 5 else base


def en_tokenizer(text):
    words = [word for word in text.split() if word[0] != "@"]  # remove mentions
    text = basic_preprocess(" ".join(words))  # basic preprocess
    words = map(lambda word: word.lower(), word_tokenize(text))  # tokenizing
    words = map(lambda word: reduce_duplicate_letters(word), words)  # remove duplicates
    words = [word for word in words if word not in stopwords.words("english")]  # remove stopwords
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))  # stemming
    # tokens = (list(map(lambda token: WordNetLemmatizer().lemmatize("rocks"), tokens)))  # lemmatize
    filtered_tokens = [token for token in tokens if len(token) >= min_term_length]  # min. length filter
    return filtered_tokens


def tr_tokenizer(text):
    text = basic_preprocess(text)
    tokens = str(text).split()
    tokens = [tr_informal[token] if token in tr_informal else token for token in tokens]  # informal to formal conv.
    words = map(lambda word: word.lower(), word_tokenize(' '.join(tokens)))  # tokenizing
    words = map(lambda word: reduce_duplicate_letters(word), words)  # remove duplicates
    words = [word for word in words if word not in stopwords.words("turkish")]  # remove stopwords
    tokens = (list(map(lambda token: fixed_prefix_stemming(token), words)))  # stemming
    filtered_tokens = [token for token in tokens if len(token) >= min_term_length]  # min. length filter
    return filtered_tokens


vec_helpers = [
    # ('bow_tf-idf', TfidfVectorizer(encoding="utf-8", min_df=mdf, tokenizer=en_tokenizer)),
    # ('bow_tf', CountVectorizer(encoding="utf-8", min_df=mdf, tokenizer=en_tokenizer)),
    ('bow_binary', CountVectorizer(encoding="utf-8", min_df=mdf, tokenizer=en_tokenizer, binary=True))
]


def get_gender_data():
    path = "../data/1/data.xlsx"
    data_frame = pd.read_excel(path)
    texts = data_frame["TXT"]
    targets = data_frame["LBL"]
    return np.array(texts), np.array(targets)


def get_pan_dataset():
    path = "../data/2/new_train.xlsx"
    df1 = pd.read_excel(path)
    df1['TXT'] = df1['TXT'].astype(str)
    path = "../data/2/new_test.xlsx"
    df2 = pd.read_excel(path)
    df2['TXT'] = df2['TXT'].astype(str)
    return df1["TXT"], df2["TXT"], df1["LBL"], df2["LBL"]


def cv_classification_without_fs(texts, targets):
    print("CV classification without feature selection:")
    for v_name, v_helper in vec_helpers:
        print("Vec_helper: {}".format(v_name))
        cv = StratifiedKFold(n_splits=fold, random_state=rs, shuffle=True)
        scores = {}
        f = 1
        for train_index, test_index in cv.split(texts, targets):
            x_train = v_helper.fit_transform(texts[train_index])
            x_train = x_train.toarray()
            x_test = v_helper.transform(texts[test_index])
            x_test = x_test.toarray()
            y_train, y_test = targets[train_index], targets[test_index]
            if partition_membership:
                if f == 1:
                    print("PM Filter is active!")
                x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
            s = x_train.shape
            print("Fold: {}- # of instances: {}, features: {}".format(str(f), str(s[0]), str(s[1])))
            for name, model in classifiers:
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
                if name not in scores:
                    scores[name] = [f_macro]
                else:
                    scores[name].append(f_macro)
                f += 1

        for k, v in scores.items():
            print("Average f-macro of " + str(k) + " is: ", mean(scores[k]))
            print("F-macro scores across " + str(fold) + " folds for " + str(k) + ":", str(scores[k]))
        print('-' * 100)


def hold_out_classification_without_fs(texts, targets):
    print("Classifying without feature selection:")
    for v_name, v_helper in vec_helpers:
        print("Vec_helper: {}".format(v_name))
        x_train, x_test, y_train, y_test = train_test_split(texts, targets, test_size=0.2, random_state=rs)
        x_train = v_helper.fit_transform(x_train)
        # print(v_helper.get_feature_names_out())
        x_train = x_train.toarray()
        x_test = v_helper.transform(x_test)
        x_test = x_test.toarray()
        if partition_membership:
            print("PM Filter is active!")
            x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
        s = x_train.shape
        print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
        for name, model in classifiers:
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def hold_out_classification_without_fs_2(train, test, train_lbl, test_lbl):
    print("Classifying without feature selection:")
    for v_name, v_helper in vec_helpers:
        print("Vec_helper: {}".format(v_name))
        x_train = v_helper.fit_transform(train)
        x_train = x_train.toarray()
        x_test = v_helper.transform(test)
        x_test = x_test.toarray()
        if partition_membership:
            print("PM Filter is active!")
            x_train, x_test, train_lbl, test_lbl = pm_filter(x_train, x_test, train_lbl, test_lbl)
        s = x_train.shape
        print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
        for name, model in classifiers:
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, train_lbl)
            predictions = model.predict(x_test)
            f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def pm_filter(x_train, x_test, y_train, y_test):
    a, b = y_train.copy(), y_test.copy()
    a, b = to_nominal_labels(a), to_nominal_labels(b)
    pm = WekaTransformer(classname=filter_name, options=filter_options)
    x_train, _ = pm.fit(x_train, a).transform(x_train, a)
    x_test, _ = pm.transform(x_test, b)
    return x_train, x_test, y_train, y_test


def hold_out_classification_with_fs(texts, targets):
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    for v_name, v_helper in vec_helpers:
        # if v_name == "bow_binary":
            idx = 0
            x_train, x_test, y_train, y_test = train_test_split(texts, targets, test_size=0.2, random_state=rs)
            print("Vec_helper: {}".format(v_name))
            x_train = v_helper.fit_transform(x_train).toarray()
            x_test = v_helper.transform(x_test).toarray()
            if partition_membership:
                print("PM Filter is active!")
                var_threshold = VarianceThreshold(threshold=0)
                x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
                s = x_train.shape
                print("# of features before variance threshold: {}".format(str(s[1])))
                x_train = var_threshold.fit_transform(x_train)
                x_test = var_threshold.transform(x_test)
            s = x_train.shape
            print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
            for k in range(10, s[1], 10):
                selector = SelectKBest(f_classif, k=k)
                x_train_new = selector.fit_transform(x_train, y_train)
                x_test_new = selector.transform(x_test)
                for name, c in classifiers:
                    c.fit(x_train_new, y_train)
                    predictions = c.predict(x_test_new)
                    f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
                    df.at[idx, "k"] = k
                    df.at[idx, name] = f_macro
                idx += 1
            print('-' * 100)
            if partition_membership:
                df.to_excel(v_name + "_pm_af.xlsx")
            else:
                df.to_excel(v_name + "_af.xlsx")

            # ==== to get only actual and predicted labels for statistical test ====
            # selector = SelectKBest(f_classif, k=750)
            # x_train_new = selector.fit_transform(x_train, y_train)
            # x_test_new = selector.transform(x_test)
            # for name, c in classifiers:
            #     if name == "LR":
            #         c.fit(x_train_new, y_train)
            #         predictions = c.predict(x_test_new)
            #         print("Actual labels: {}".format(list(y_test)))
            #         print("Predicted labels: {}".format(predictions))
            #         f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
            #         print("Avg. f_macro: {}".format(f_macro))


def hold_out_classification_with_fs_2(train, test, train_lbl, test_lbl):
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    for v_name, v_helper in vec_helpers:
        idx = 0
        print("Vec_helper: {}".format(v_name))
        x_train = v_helper.fit_transform(train).toarray()
        x_test = v_helper.transform(test).toarray()
        if partition_membership:
            print("PM Filter is active!")
            var_threshold = VarianceThreshold(threshold=0)
            x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
            s = x_train.shape
            print("# of features before variance threshold: {}".format(str(s[1])))
            x_train = var_threshold.fit_transform(x_train)
            x_test = var_threshold.transform(x_test)
        s = x_train.shape
        print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
        for k in range(10, s[1], 10):
            print("{}/{}".format(str(k), s[1]))
            selector = SelectKBest(f_classif, k=k)
            x_train_new = selector.fit_transform(x_train, train_lbl)
            x_test_new = selector.transform(x_test)
            for name, c in classifiers:
                c.fit(x_train_new, train_lbl)
                predictions = c.predict(x_test_new)
                f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
                df.at[idx, "k"] = k
                df.at[idx, name] = f_macro
            idx += 1
            if k % 1000 == 0:
                df.to_excel(v_name + "_" + str(k) + "_pm_af.xlsx")
        print('-' * 100)
        if partition_membership:
            df.to_excel(v_name + "_pm_af.xlsx")
        else:
            df.to_excel(v_name + "_af.xlsx")


def classify_with_fasttext_vectors(texts, targets, lang):
    doc = []
    texts = list(texts)
    if lang == "tr":
        cap_path = datapath("C:/Users/Onder/Desktop/cc.tr.300.bin.gz")
        for i in range(len(texts)):
            doc.insert(i, ' '.join(tr_tokenizer(texts[i])))
    else:
        cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
        for i in range(len(texts)):
            doc.insert(i, ' '.join(en_tokenizer(texts[i])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    vectors = []
    for j in range(len(doc)):
        words = doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        vectors.insert(j, avg)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    vectors = np.array(vectors)
    x_train, x_test, y_train, y_test = train_test_split(vectors, targets, test_size=0.2, random_state=rs)
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
    s = x_train.shape
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for name, model in classifiers:
        # if name == "RF":
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            print("Actual labels: {}".format(list(y_test)))
            print("Predicted labels: {}".format(predictions))
            f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def classify_with_fasttext_vectors_2(train, test, y_train, y_test):
    train_doc, test_doc = [], []
    cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
    for i in range(len(train)):
        train_doc.insert(i, ' '.join(en_tokenizer(train[i])))
    for i in range(len(test)):
        test_doc.insert(i, ' '.join(en_tokenizer(test[i])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    train_vectors, test_vectors = [], []
    for j in range(len(train_doc)):
        words = train_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        train_vectors.insert(j, avg)
    x_train = np.array(train_vectors)
    for j in range(len(test_doc)):
        words = test_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        test_vectors.insert(j, avg)
    x_test = np.array(test_vectors)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
    s = x_train.shape
    ss = x_test.shape
    print("# of train instances: {}, features: {}".format(str(s[0]), str(s[1])))
    print("# of test instances: {}, features: {}".format(str(ss[0]), str(ss[1])))
    for name, model in classifiers:
        if name == "SVM":
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            print("Actual labels: {}".format(list(y_test)))
            print("Predicted labels: {}".format(predictions))
            f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def knn_variants_classification_with_fasttext_vectors(train, test, train_lbl, test_lbl):
    train_doc, test_doc = [], []
    cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
    for i in range(len(train)):
        train_doc.insert(i, ' '.join(en_tokenizer(train[i])))
    for i in range(len(test)):
        test_doc.insert(i, ' '.join(en_tokenizer(test[i])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    train_vectors, test_vectors = [], []
    for j in range(len(train_doc)):
        words = train_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        train_vectors.insert(j, avg)
    x_train = np.array(train_vectors)
    for j in range(len(test_doc)):
        words = test_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        test_vectors.insert(j, avg)
    x_test = np.array(test_vectors)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
    s = x_train.shape
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    classifiers = [
        ('KNN_MIN', KNeighborsClassifier(n_neighbors=1, metric='minkowski')),
        ('KNN_EUC', KNeighborsClassifier(n_neighbors=1, metric='euclidean')),
        ('KNN_CHE', KNeighborsClassifier(n_neighbors=1, metric='chebyshev')),
    ]
    for name, model in classifiers:
        print("Hold-out classification with " + name + "..")
        model.fit(x_train, train_lbl)
        predictions = model.predict(x_test)
        f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
        print("Average f-macro of " + name + " is: ", f_macro)
        print('-' * 100)


def knn_variants_classification_with_bert_without_fs_2(train, test, train_lbl, test_lbl):
    train_docs, test_docs = [], []
    for j in range(len(train)):
        train_docs.insert(j, " ".join(en_tokenizer(train[j])))
    for j in range(len(test)):
        test_docs.insert(j, " ".join(en_tokenizer(test[j])))
    train_vectors = get_bert_embeddings(train_docs)
    test_vectors = get_bert_embeddings(test_docs)
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
    s = x_train.shape
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    classifiers = [
        ('KNN_MIN', KNeighborsClassifier(n_neighbors=1, metric='minkowski')),
        ('KNN_EUC', KNeighborsClassifier(n_neighbors=1, metric='euclidean')),
        ('KNN_CHE', KNeighborsClassifier(n_neighbors=1, metric='chebyshev')),
    ]
    for name, model in classifiers:
        print("Hold-out classification with " + name + "..")
        model.fit(x_train, train_lbl)
        predictions = model.predict(x_test)
        f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
        print("Average f-macro of " + name + " is: ", f_macro)
        print('-' * 100)


def classify_with_fasttext_vectors_with_fs(texts, targets, lang):
    doc = []
    texts = list(texts)
    if lang == "tr":
        cap_path = datapath("C:/Users/Onder/Desktop/cc.tr.300.bin.gz")
        for i in range(len(texts)):
            doc.insert(i, ' '.join(tr_tokenizer(texts[i])))
    else:
        cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
        for i in range(len(texts)):
            doc.insert(i, ' '.join(en_tokenizer(texts[i])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    vectors = []
    for j in range(len(doc)):
        words = doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        vectors.insert(j, avg)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    vectors = np.array(vectors)
    x_train, x_test, y_train, y_test = train_test_split(vectors, targets, test_size=0.2, random_state=rs)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for k in range(10, s[1], 10):
        selector = SelectKBest(f_classif, k=k)
        x_train_new = selector.fit_transform(x_train, y_train)
        x_test_new = selector.transform(x_test)
        for name, c in classifiers:
            if name != "MNB":
                c.fit(x_train_new, y_train)
                predictions = c.predict(x_test_new)
                f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
                df.at[idx, "k"] = k
                df.at[idx, name] = f_macro
        idx += 1
    print('-' * 100)
    if partition_membership:
        df.to_excel("fasttext_pm_af.xlsx")
    else:
        df.to_excel("fasttext_af.xlsx")
    # ==== for getting only actual and predictions for statistical testing ====
    # selector = SelectKBest(f_classif, k=50)
    # x_train_new = selector.fit_transform(x_train, y_train)
    # x_test_new = selector.transform(x_test)
    # for name, c in classifiers:
    #     if name == "RF":
    #         c.fit(x_train_new, y_train)
    #         predictions = c.predict(x_test_new)
    #         print("Actual labels: {}".format(list(y_test)))
    #         print("Predicted labels: {}".format(predictions))
    #         f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
    #         print("Avg. f_macro: {}".format(f_macro))


def classify_with_fasttext_vectors_with_fs_2(train, test, train_lbl, test_lbl):
    train_doc, test_doc = [], []
    cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
    for i in range(len(train)):
        train_doc.insert(i, ' '.join(en_tokenizer(train[i])))
    for j in range(len(test)):
        test_doc.insert(j, ' '.join(en_tokenizer(test[j])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    train_vectors, test_vectors = [], []
    for j in range(len(train_doc)):
        words = train_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        train_vectors.insert(j, avg)
    for j in range(len(test_doc)):
        words = test_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        test_vectors.insert(j, avg)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for k in range(10, s[1], 10):
        selector = SelectKBest(f_classif, k=k)
        x_train_new = selector.fit_transform(x_train, train_lbl)
        x_test_new = selector.transform(x_test)
        for name, c in classifiers:
            # if name != "MNB":
                c.fit(x_train_new, train_lbl)
                predictions = c.predict(x_test_new)
                f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
                df.at[idx, "k"] = k
                df.at[idx, name] = f_macro
        idx += 1
    print('-' * 100)
    if partition_membership:
        df.to_excel("fasttext_pm_af.xlsx")
    else:
        df.to_excel("fasttext_af.xlsx")


def knn_variants_classification_with_fasttext_vectors_with_fs_2(train, test, train_lbl, test_lbl):
    train_doc, test_doc = [], []
    cap_path = datapath("C:/Users/Onder/Desktop/cc.en.300.bin.gz")
    for i in range(len(train)):
        train_doc.insert(i, ' '.join(en_tokenizer(train[i])))
    for j in range(len(test)):
        test_doc.insert(j, ' '.join(en_tokenizer(test[j])))

    wv_model = load_facebook_vectors(cap_path)
    print("vector size: " + str(wv_model.vector_size))
    num_found = 0
    num_not_found = 0
    train_vectors, test_vectors = [], []
    for j in range(len(train_doc)):
        words = train_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        train_vectors.insert(j, avg)
    for j in range(len(test_doc)):
        words = test_doc[j].split()
        avg = [np.zeros(wv_model.vector_size)]
        for word in words:
            t_vector = wv_model.get_vector(word)
            if t_vector is None:
                word = word.capitalize()
                t_vector = wv_model.get_vector(word)
                if t_vector is not None:
                    avg.append(t_vector)
                    num_found = num_found + 1
                else:
                    num_not_found += 1
            else:
                avg.append(t_vector)
                num_found = num_found + 1
        avg = np.array(avg)
        avg = np.mean(avg, axis=0)
        test_vectors.insert(j, avg)
    print("Total # of terms found and not-found in model: {}, {}".format(num_found, num_not_found))
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    classifiers = [
        ('KNN_MIN', KNeighborsClassifier(n_neighbors=1, metric='minkowski')),
        ('KNN_EUC', KNeighborsClassifier(n_neighbors=1, metric='euclidean')),
        ('KNN_CHE', KNeighborsClassifier(n_neighbors=1, metric='chebyshev')),
    ]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for k in range(10, s[1], 10):
        selector = SelectKBest(f_classif, k=k)
        x_train_new = selector.fit_transform(x_train, train_lbl)
        x_test_new = selector.transform(x_test)
        for name, c in classifiers:
            c.fit(x_train_new, train_lbl)
            predictions = c.predict(x_test_new)
            f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
            df.at[idx, "k"] = k
            df.at[idx, name] = f_macro
        idx += 1
    print('-' * 100)
    if partition_membership:
        df.to_excel("fasttext_pm_af.xlsx")
    else:
        df.to_excel("fasttext_af.xlsx")


def get_bert_embeddings(texts):
    # llm = "dbmdz/bert-base-turkish-cased"
    # llm = "hemekci/off_detection_turkish"
    # llm = "dbmdz/bert-base-turkish-uncased"
    # llm = "dbmdz/bert-base-turkish-128k-cased"

    # llm = "bert-large-uncased"
    # llm = "bert-base-uncased"
    llm = "thaile/bert-base-uncased-md_gender_bias-saved"
    # llm = "Abderrahim2/bert-finetuned-gender_classification"

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rs)
    tokenizer = BertTokenizer.from_pretrained(llm)
    model = BertModel.from_pretrained(llm)
    vectors = []
    for i in range(len(texts)):
        text = texts[i]
        encoding = tokenizer.batch_encode_plus([text], padding=True, truncation=True,
                                               return_tensors='pt', add_special_tokens=True)
        input_ids = encoding['input_ids']  # Token IDs
        # print(f"Input ID: {input_ids}")
        attention_mask = encoding['attention_mask']  # Attention mask
        # print(f"Attention mask: {attention_mask}")
        # Generate embeddings using BERT model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = outputs.last_hidden_state  # This contains the embeddings
        # Output the shape of word embeddings
        # print(f"Shape of Word Embeddings: {word_embeddings.shape}")
        # Decode the token IDs back to text
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        tokenized_text = tokenizer.tokenize(decoded_text)
        encoded_text = tokenizer.encode(text, return_tensors='pt')  # Returns a tensor
        # Compute the average of word embeddings to get the sentence embedding
        sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension
        # print(f"Shape of Sentence Embedding: {sentence_embedding.shape}")
        x = sentence_embedding.numpy()
        x = list(x)
        vectors.insert(i, x[0])
    return vectors


def classify_with_bert_without_fs(texts, targets, lang):
    if lang == "tr":
        d = []
        for j in range(len(texts)):
            d.insert(j, " ".join(tr_tokenizer(texts[j])))
        texts = d
    else:
        d = []
        for j in range(len(texts)):
            d.insert(j, " ".join(en_tokenizer(texts[j])))
        texts = d
    vectors = get_bert_embeddings(texts)
    vectors = np.array(vectors)
    x_train, x_test, y_train, y_test = train_test_split(vectors, targets, test_size=0.2, random_state=rs)
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
    s = x_train.shape
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for name, model in classifiers:
        # if name != "MNB":
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def classify_with_bert_without_fs_2(train, test, train_lbl, test_lbl):
    train_docs, test_docs = [], []
    for j in range(len(train)):
        train_docs.insert(j, " ".join(en_tokenizer(train[j])))
    for j in range(len(test)):
        test_docs.insert(j, " ".join(en_tokenizer(test[j])))
    train_vectors = get_bert_embeddings(train_docs)
    test_vectors = get_bert_embeddings(test_docs)
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
    s = x_train.shape
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for name, model in classifiers:
        if name == "MNB":
            print("Hold-out classification with " + name + "..")
            model.fit(x_train, train_lbl)
            predictions = model.predict(x_test)
            print("Actual labels: {}".format(list(test_lbl)))
            print("Predicted labels: {}".format(predictions))
            f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
            print("Average f-macro of " + name + " is: ", f_macro)
            print('-' * 100)


def classify_with_bert_with_fs(texts, targets, lang):
    if lang == "tr":
        d = []
        for j in range(len(texts)):
            d.insert(j, " ".join(tr_tokenizer(texts[j])))
        texts = d
    else:
        d = []
        for j in range(len(texts)):
            d.insert(j, " ".join(en_tokenizer(texts[j])))
        texts = d
    vectors = get_bert_embeddings(texts)
    vectors = np.array(vectors)
    x_train, x_test, y_train, y_test = train_test_split(vectors, targets, test_size=0.2, random_state=rs)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, y_train, y_test)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for k in range(10, s[1], 10):
        selector = SelectKBest(f_classif, k=k)
        x_train_new = selector.fit_transform(x_train, y_train)
        x_test_new = selector.transform(x_test)
        for name, c in classifiers:
            # if name != "MNB":
                c.fit(x_train_new, y_train)
                predictions = c.predict(x_test_new)
                f_macro = metrics.f1_score(y_true=y_test, y_pred=predictions, average="macro")
                df.at[idx, "k"] = k
                df.at[idx, name] = f_macro
        idx += 1
    print('-' * 100)
    if partition_membership:
        df.to_excel("bert_pm_af.xlsx")
    else:
        df.to_excel("bert_af.xlsx")


def classify_with_bert_with_fs_2(train, test, train_lbl, test_lbl):
    train_docs, test_docs = [], []
    for j in range(len(train)):
        train_docs.insert(j, " ".join(en_tokenizer(train[j])))
    for i in range(len(test)):
        test_docs.insert(i, ' '.join(en_tokenizer(test[i])))
    train_vectors = get_bert_embeddings(train_docs)
    test_vectors = get_bert_embeddings(test_docs)
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    # for k in range(10, s[1], 10):
    #     selector = SelectKBest(f_classif, k=k)
    #     x_train_new = selector.fit_transform(x_train, train_lbl)
    #     x_test_new = selector.transform(x_test)
    #     for name, c in classifiers:
    #         # if name != "MNB":
    #             c.fit(x_train_new, train_lbl)
    #             predictions = c.predict(x_test_new)
    #             f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
    #             df.at[idx, "k"] = k
    #             df.at[idx, name] = f_macro
    #     idx += 1
    # print('-' * 100)
    # if partition_membership:
    #     df.to_excel("bert_pm_af.xlsx")
    # else:
    #     df.to_excel("bert_af.xlsx")
    # ==== for getting only actual and predicted labels for statistical testing
    selector = SelectKBest(f_classif, k=40)
    x_train_new = selector.fit_transform(x_train, train_lbl)
    x_test_new = selector.transform(x_test)
    for name, c in classifiers:
        if name == "SVM":
            c.fit(x_train_new, train_lbl)
            predictions = c.predict(x_test_new)
            print("Actual labels: {}".format(list(test_lbl)))
            print("Predicted label: {}".format(predictions))
            f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
            print("Avg. f_macro: {}".format(f_macro))


def knn_variants_classification_with_bert_with_fs_2(train, test, train_lbl, test_lbl):
    train_docs, test_docs = [], []
    for j in range(len(train)):
        train_docs.insert(j, " ".join(en_tokenizer(train[j])))
    for i in range(len(test)):
        test_docs.insert(i, ' '.join(en_tokenizer(test[i])))
    train_vectors = get_bert_embeddings(train_docs)
    test_vectors = get_bert_embeddings(test_docs)
    x_train = np.array(train_vectors)
    x_test = np.array(test_vectors)
    if partition_membership:
        print("PM Filter is active!")
        var_threshold = VarianceThreshold(threshold=0)
        x_train, x_test, y_train, y_test = pm_filter(x_train, x_test, train_lbl, test_lbl)
        s = x_train.shape
        print("# of features before variance threshold: {}".format(str(s[1])))
        x_train = var_threshold.fit_transform(x_train)
        x_test = var_threshold.transform(x_test)
    s = x_train.shape
    columns = ["k"]
    classifiers = [
        ('KNN_MIN', KNeighborsClassifier(n_neighbors=1, metric='minkowski')),
        ('KNN_EUC', KNeighborsClassifier(n_neighbors=1, metric='euclidean')),
        ('KNN_CHE', KNeighborsClassifier(n_neighbors=1, metric='chebyshev')),
    ]
    for k, v in classifiers:
        columns.append(k)
    df = pd.DataFrame(columns=columns)
    idx = 0
    print("# of instances: {}, features: {}".format(str(s[0]), str(s[1])))
    for k in range(10, s[1], 10):
        selector = SelectKBest(f_classif, k=k)
        x_train_new = selector.fit_transform(x_train, train_lbl)
        x_test_new = selector.transform(x_test)
        for name, c in classifiers:
            c.fit(x_train_new, train_lbl)
            predictions = c.predict(x_test_new)
            f_macro = metrics.f1_score(y_true=test_lbl, y_pred=predictions, average="macro")
            df.at[idx, "k"] = k
            df.at[idx, name] = f_macro
        idx += 1
    print('-' * 100)
    if partition_membership:
        df.to_excel("bert_pm_af.xlsx")
    else:
        df.to_excel("bert_af.xlsx")


def toy_scenario_for_pmf():
    train_vectors = [
        [0.0710, 0.0048, -0.3006, 0.0629, 0.0114],
        [0.0824, 0.0232, 0.2506, 0.0233, -0.0343],
        [0.0515, 0.0458, -0.2116, -0.1320, -0.1820],
        [0.0761, -0.0659, -0.5315, 0.0471, 0.0824],
        [-0.0494, 0.1818, 0.0318, -0.0678, -0.2771]
    ]
    x_train = np.array(train_vectors)
    # var_threshold = VarianceThreshold(threshold=0)
    y_train = [0, 0, 1, 1, 1]
    a = to_nominal_labels(y_train)
    pm = WekaTransformer(classname=filter_name, options=filter_options)
    x_train, _ = pm.fit(x_train, a).transform(x_train, a)
    s = x_train.shape
    print("# of features before variance threshold: {}".format(str(s[1])))
    # x_train = var_threshold.fit_transform(x_train)
    print(x_train.shape)
    print(x_train)


if __name__ == "__main__":
    try:
        jvm.start()
        # docs, labels = get_r8_data()
        # ============== EN ============================================================
        train, test, train_lbl, test_lbl = get_pan_dataset()
        # hold_out_classification_without_fs_2(train, test, train_lbl, test_lbl)
        # hold_out_classification_with_fs_2(train, test, train_lbl, test_lbl)
        # classify_with_fasttext_vectors_with_fs_2(train, test, train_lbl, test_lbl)
        classify_with_bert_with_fs_2(train, test, train_lbl, test_lbl)
        # classify_with_fasttext_vectors_2(train, test, train_lbl, test_lbl)
        # classify_with_bert_without_fs_2(train, test, train_lbl, test_lbl)

        # ============== TR ============================================================
        # docs, labels = get_gender_data()
        # cv_classification_without_fs(texts=docs, targets=labels)
        # hold_out_classification_without_fs(texts=docs, targets=labels)
        # hold_out_classification_with_fs(texts=docs, targets=labels)
        # classify_with_fasttext_vectors(texts=docs, targets=labels, lang="tr")
        # classify_with_fasttext_vectors_with_fs(texts=docs, targets=labels, lang="tr")
        # classify_with_bert_without_fs(texts=docs, targets=labels, lang="tr")
        # classify_with_bert_with_fs(texts=docs, targets=labels, lang="tr")

        # ============== KNN - EN =======================================================
        # knn_variants_classification_with_fasttext_vectors(train, test, train_lbl, test_lbl)
        # knn_variants_classification_with_bert_without_fs_2(train, test, train_lbl, test_lbl)
        # knn_variants_classification_with_fasttext_vectors_with_fs_2(train, test, train_lbl, test_lbl)
        # knn_variants_classification_with_bert_with_fs_2(train, test, train_lbl, test_lbl)

        # ============== TOY SCENARIO =======================================================
        # toy_scenario_for_pmf()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
