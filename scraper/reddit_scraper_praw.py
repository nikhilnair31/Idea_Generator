import re
import praw
import json
import pandas as pd
from sklearn.model_selection import train_test_split

key_path = 'keys/reddit_keys.json'
csv_path = './data/csv/reddit_gameideas_top_all_100.csv'
train_txt_path = './data/txt/reddit_gameideas_train_tot.txt'
test_txt_path = './data/txt/reddit_gameideas_test_tot.txt'
posts = []
subreddit = 'gameideas'
limit = 1000
split_ratio = 0.05

def clean_text(text):
    rem_text = text.strip('"')
    rem_text = rem_text.replace("\n", "")
    if (re.match('^[A-Z][\w\s]+[?.!]$', rem_text) is not None):
        new = list(rem_text)
        new[-1] = '.'
        rem_text = ''.join(new)
    return rem_text

def to_df_csv(posts):
    posts = pd.DataFrame(posts, columns=['title+desc'])
    posts.to_csv(csv_path)

def to_one_txt_file(stringlist, filepath):
    for one_string in stringlist:
        f = open(filepath, 'a', encoding='utf-8')
        f.write( str(one_string) )
        f.close()

def scrapey(gd_reddit):
    top_posts = gd_reddit.subreddit(subreddit)
    for post in top_posts.top("all", limit=limit):
        cleaned_title = clean_text(post.title)
        cleaned_desc = clean_text(post.selftext)
        if(cleaned_desc != ''):
            cleaned_desc += '\n'
        posts.append('<|title|>' + cleaned_title + cleaned_desc + '<|endoftext|>')
    
    to_df_csv(posts)
    train, test = train_test_split(posts, test_size = split_ratio)
    to_one_txt_file(train, train_txt_path)
    to_one_txt_file(test, test_txt_path)

def plotr():
    df = pd.read_csv(csv_path) 
    print(df.shape)
    df['word_count'] = df['title+desc'].apply(lambda x: len(str(x).split()))
    df['word_count'].plot( kind='hist',  bins = 50, figsize = (12,8),title='Word Count Distribution')

if __name__=="__main__":
    with open(key_path) as f:
        data = json.load(f)
    client_id = data["client_id"]
    client_secret = data["client_secret"]

    gd_reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=subreddit)
    scrapey(gd_reddit)
    plotr()