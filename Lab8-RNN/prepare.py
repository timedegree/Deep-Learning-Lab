from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import nltk
import os
import re

nltk.download('punkt_tab')

tqdm.pandas()
pattern = re.compile(r"^data/aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt$")

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
            match = pattern.match(file)
            comments["comments"] = content
            comments["sentiment"] = 1 if match.group(2) == "pos" in file else 0
            comments["score"] = match.group(3) if match else None
        comments_list.append(comments)

    return comments_list

def clean_text(text):
    # Remove links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Add space between certain punctuation
    text = re.sub(r'([.,!?-])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove selected punctuation
    text = re.sub(r'[\"\#\$\%\&\'\(\)\*\+\/\:\;\<\=\>\@\[\\\]\^\_\`\{\|\}\~]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    # Remove emojis
    text = re.sub(
        r'['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F1E0-\U0001F1FF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        '', text, flags=re.UNICODE
    )
    # Basic spell correction (e.g., reduce repeated chars)
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    return text.strip()

# preprocessing
def tokenize(text):
    return word_tokenize(text)

def rm_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return [i for i in text if i not in stop_words]

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()    
    lemmas = [lemmatizer.lemmatize(t) for t in text]
    # make sure lemmas does not contains sotpwords
    return rm_stopwords(lemmas)

def preprocess_pipeline(text):
    tokens = tokenize(text)
    no_stopwords = rm_stopwords(tokens)
    lemmas = lemmatize(no_stopwords)
    return ' '.join(lemmas)


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

train_data['cleaned_comments'] = train_data['comments'].progress_apply(clean_text)
test_data['cleaned_comments'] = test_data['comments'].progress_apply(clean_text)
train_data['preprocessed_comments'] = train_data['cleaned_comments'].progress_apply(preprocess_pipeline)
test_data['preprocessed_comments'] = test_data['cleaned_comments'].progress_apply(preprocess_pipeline)
train_data = train_data[['comments','preprocessed_comments', 'sentiment', 'score']]
test_data = test_data[['comments','preprocessed_comments', 'sentiment', 'score']]

train_data.to_csv('data/IMDB_train.csv', index=False)
test_data.to_csv('data/IMDB_test.csv', index=False)