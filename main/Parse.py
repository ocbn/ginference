import os
from bs4 import BeautifulSoup
import pandas as pd


def parse():
    d_type = "test"
    path = "C:/Users/Onder/Desktop/pan18_" + d_type
    file_list_dir = path + "/en/" + "text"
    file_names = path + "/en/en.txt"
    file_list = os.listdir(file_list_dir)

    cat_dict = {}
    f1 = open(file_names, "r")
    lines = f1.readlines()
    for line in lines:
        parts = line.strip().rstrip().split(":::")
        f_name = parts[0]
        label = parts[1]
        cat_dict[f_name] = label

    print("# of samples: " + str(len(cat_dict)))

    df = pd.DataFrame(columns=["TXT", "LBL"])
    num_samples = 0
    num_docs = 0
    idx = 0
    for sample in file_list:
        print("{}-{}".format(num_samples, len(file_list)))
        num_samples += 1
        f2 = open(file_list_dir + "/" + sample, "r", encoding="utf8")
        data = f2.read()
        bs_data = BeautifulSoup(data, "xml")
        docs = bs_data.find_all('document')
        for doc in docs:
            num_docs += 1
            content = str(doc.contents[0]).strip().rstrip()
            lbl = cat_dict[sample.split(".")[0]]
            df.at[idx, "TXT"] = content
            df.at[idx, "LBL"] = lbl
            idx += 1
    print("# of docs: " + str(num_docs))
    print("# of samples/files: " + str(num_samples))
    df.to_excel(d_type + ".xlsx", index=False)


def sample():
    p = "test"
    path = "C:/Users/Onder/Desktop/" + p + ".xlsx"
    df = pd.read_excel(path)
    df['TXT'] = df['TXT'].apply(lambda x: 'NONE' if len(str(x).split()) < 20 else x)
    df1 = df[(df['TXT'] != 'NONE') & (df['LBL'] == "female")].sample(n=500)
    df0 = df[(df['TXT'] != 'NONE') & (df['LBL'] == "male")].sample(n=500)
    df = pd.concat([df1, df0])
    df['LBL'] = df['LBL'].apply(lambda x: 1 if str(x) == "female" else 0)
    df.to_excel("new_" + p + ".xlsx", index=False)


sample()
# parse_2()
