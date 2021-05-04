import re
import praw
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    rem_text = text.strip('"')
    rem_text = rem_text.replace("\n", "")
    if (re.match('^[A-Z][\w\s]+[?.!]$', rem_text) is not None):
      new = list(rem_text)
      new[-1] = '.'
      rem_text = ''.join(new)
      # print('post rem_text: ',rem_text)
    return rem_text

def to_df_csv(posts):
    posts = pd.DataFrame(posts, columns=['title+body'])
    posts.to_csv('./data/reddit/csv/reddit_gameideas_top_all_100.csv')

def to_one_txt_file(stringlist, filepath):
    for one_string in stringlist:
        f = open(filepath, 'a', encoding='utf-8')
        f.write( str(one_string) )
        f.close()

def scrapey(gd_reddit, posts, limit, split_ratio):
    top_posts = gd_reddit.subreddit('gameideas')
    for post in top_posts.top("all", limit=limit):
        # posts.append([ remove_line_breaks(post.title), remove_line_breaks(post.selftext), post.score, post.num_comments ])
        cleaned_title = clean_text(post.title)
        cleaned_descr = clean_text(post.selftext)
        if(cleaned_descr != ''):
            cleaned_descr += '\n'
        posts.append( cleaned_title + ' \n' + cleaned_descr + ' \n' )
        # posts.append( cleaned_title + ' \n')
    
    print('pre posts[:3]: ',posts[:3])
    new_posts = [' '.join(item.split(' ')[:512]) for item in posts if item]
    print('post new_posts[:3]: ',new_posts[:3])
    to_df_csv(posts)
    train, test = train_test_split(posts, test_size = split_ratio)
    to_one_txt_file(train, './data/reddit/txt/reddit_gameideas_train_tot.txt')
    to_one_txt_file(test, './data/reddit/txt/reddit_gameideas_test_tot.txt')

def plotr():
    df = pd.read_csv("./data/reddit/csv/reddit_gameideas_top_all_100.csv") 
    print(df.shape)
    df['word_count'] = df['title+body'].apply(lambda x: len(str(x).split()))
    df['word_count'].plot( kind='hist',  bins = 50, figsize = (12,8),title='Word Count Distribution')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='You can add a description here')
    parser.add_argument('-limit','--limit', help='limit', type=int, required=True)
    parser.add_argument('-split','--split_ratio', help='split_ratio', type=float, required=True)
    parser.add_argument('-sub','--subreddit_to_scrape', help='subreddit_to_scrape', type=str, required=True)
    args = parser.parse_args()

    posts = []
    client_id = 'hRWv3owMNdlZGg'
    client_secret='2Ojet5wqaMRn3x4kJSZPO0e0Ae4VEw'
    gd_reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=args.subreddit_to_scrape)
    scrapey(gd_reddit, posts, args.limit, args.split_ratio)
    plotr()