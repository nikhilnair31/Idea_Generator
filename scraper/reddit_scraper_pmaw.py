import re
import datetime as dt
from pmaw import PushshiftAPI
import pandas as pd
from sklearn.model_selection import train_test_split

posts = []
subreddit = 'gameideas'
limit = 10
split_ratio = 0.05
before = int(dt.datetime(2021,5,1,0,0).timestamp())
after = int(dt.datetime(2016,5,1,0,0).timestamp())
key_path = 'keys/reddit_keys.json'
csv_path = f'./data/csv/reddit_{subreddit}_{limit}_{before}_{after}.csv'
train_txt_path = f'./data/txt/reddit_{subreddit}_{limit}_{before}_{after}_train_tot.txt'
test_txt_path = f'./data/txt/reddit_{subreddit}_{limit}_{before}_{after}_test_tot.txt'

def clean_text(text):
    rem_text = text.strip('"')
    rem_text = rem_text.replace("\n", "")
    if (re.match('^[A-Z][\w\s]+[?.!]$', rem_text) is not None):
      new = list(rem_text)
      new[-1] = '.'
      rem_text = ''.join(new)
    return rem_text

def to_one_txt_file(stringlist, filepath):
    open(filepath, 'w').close()
    for one_string in stringlist:
        f = open(filepath, 'a', encoding='utf-8')
        f.write( str(one_string) )
        f.close()

def clean_to_text(submissions_df):
    for index, row in submissions_df.iterrows():
        cleaned_title = clean_text(str(row['title']))
        cleaned_desc = clean_text(str(row['selftext']))
        posts.append('<|title|>' + cleaned_title + cleaned_desc + '<|endoftext|>')
    
    train, test = train_test_split(posts, test_size = split_ratio)
    print(f'training size {len(train)}\ntest size: {len(test)}')

    to_one_txt_file(train, train_txt_path)
    to_one_txt_file(test, test_txt_path)

def plotr():
    print(submissions_df.shape)
    submissions_df['word_count'] = submissions_df['title'].apply(lambda x: len(str(x).split()))
    submissions_df['word_count'].plot( kind='hist', bins = 50, figsize = (12,8),title='Word Count Distribution')

if __name__=="__main__":
    api = PushshiftAPI()

    submissions = api.search_submissions(subreddit=subreddit, limit=limit, before=before, after=after)
    print(f'Retrieved {len(submissions)} submissions from Pushshift')

    submissions_df = pd.DataFrame(submissions)
    submissions_df.to_csv(csv_path)
    clean_to_text(submissions_df)

    plotr()