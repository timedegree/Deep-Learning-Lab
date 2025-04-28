import pandas as pd
import os
import re 

pattern = re.compile(r"^data/aclImdb/train/(pos|neg)/\d+_(\d+)\.txt$")

def get_file_path(path):
    file_path_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt') :
                file_path_list.append(os.path.join(root, file))

    return file_path_list

def read_file(file_list):
    comments_list = []

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            comments = {}
            content = f.read()
            comments["comments"] = content
            comments["sentiment"] = 1 if 'pos' in file else 0
            match = pattern.match(file)
            comments["score"] = match.group(2) if match else None
        comments_list.append(comments)

    return comments_list


train_poscomment_path = get_file_path('data/aclImdb/train/pos')
train_negcomment_path = get_file_path('data/aclImdb/train/neg')
test_poscomment_path = get_file_path('data/aclImdb/test/pos')
test_negcomment_path = get_file_path('data/aclImdb/test/neg')

print('train_poscomment_path:', train_poscomment_path[:5])
print('train_poscomment_path len:', len(train_poscomment_path))

train_comments = read_file(train_poscomment_path) + read_file(train_negcomment_path)
test_comments = read_file(test_poscomment_path) + read_file(test_negcomment_path)

train_data = pd.DataFrame(train_comments)
test_data = pd.DataFrame(test_comments)

train_data.to_csv('data/IMDB_train.csv', index=False)
test_data.to_csv('data/IMDB_test.csv', index=False)