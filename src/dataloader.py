

import pandas as pd



def load(filename):
    """ Download the date: list of texts with scores."""
    headers = ['rating', 'text']
    reviews = pd.read_csv(filename, encoding="utf-8", sep='\t', names=headers)
    # remove instances where the rating value is absent or incorrect due to incorrect format
    reviews.drop(reviews[(reviews.rating<1) | (reviews.rating>5)].index, inplace=True)
    # remove
    reviews = reviews.assign(
        text=reviews.text.map(lambda text: text.replace('\\n', '\n'))
    )
    # Convert to 3 classes: neg, neut and pos
    reviews = reviews.assign(clabel=reviews.rating.map(lambda v: 'neg' if v < 3 else 'neut' if v==3 else 'pos'))
    # print distributions by rating or class
    # print(reviews.groupby('rating').nunique())
    # print(reviews.groupby('clabel').nunique())
    # return the 2 lists : texts and labels
    return (reviews.text, reviews.clabel)



if __name__ == "__main__":
    # Just for testing
    filename = "../data/reviews_dev3K.csv"
    texts, labels = load(filename)
    print("Dataset size: %d" % len(texts))

