# Idea_Generator

## Scrape Data
* Use `reddit_scraper.py` to scrape needed subreddits. Data scraped then gets saved as txt file and split into test and train set.
* Command line parameters are: 
    * `-num 1` this is the number of returned sequences
    * `-len 32` this is the max length of a sequence
    * `-text 'Make a game'` this is the text to generate from

## GPT-3
* Use `gpt3_test.py` to test the API.

## Transformers
* Train on Google Colab with nikhilnair3118 account.
* Use `idea_gen_gpt2_test.py` to check idea generation using trained GPT-2 models from HuggingFace.