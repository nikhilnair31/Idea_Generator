import praw
import pandas as pd
from sklearn.model_selection import train_test_split

def to_df_csv():
    global posts

    posts = pd.DataFrame(posts, columns=['title', 'body', 'score', 'num_comments'])
    posts.to_csv('./data/reddit/csv/reddit_gameideas_top_all_100.csv')
    to_multiple_txt_files()

def to_one_txt_file(stringlist, filepath):
    for one_string in stringlist:
        f = open(filepath, 'a')
        f.write( one_string )
        f.close()

def scrapey(gd_reddit, posts, limit, split_ratio):
    top_posts = gd_reddit.subreddit('gameideas')
    for post in top_posts.top("all", limit=limit):
        # posts.append([ remove_line_breaks(post.title), remove_line_breaks(post.selftext), post.score, post.num_comments ])
        cleaned_title = (post.title).replace("\n", "")
        cleaned_descr = (post.selftext).replace("\n", "")
        if(cleaned_descr != ''):
            cleaned_descr += '\n'
        posts.append( cleaned_title + ' \n' + cleaned_descr + ' \n' )
    
    train, test = train_test_split(posts, test_size = split_ratio)
    to_one_txt_file(train, './data/reddit/txt/reddit_gameideas_train_tot.txt')
    to_one_txt_file(test, './data/reddit/txt/reddit_gameideas_test_tot.txt')

if __name__=="__main__":
    posts = []
    limit = 10
    split_ratio = 0.2
    gd_reddit = praw.Reddit(client_id='hRWv3owMNdlZGg', client_secret='2Ojet5wqaMRn3x4kJSZPO0e0Ae4VEw', user_agent='game ideas')
    scrapey(gd_reddit, posts, limit, split_ratio)