import os

import pandas as pd

from joblib import Parallel, delayed
from textblob import TextBlob

NAN_WORD = '_NAN_'


def translate(comment, language):
    if hasattr(comment, 'decode'):
        comment = comment.decode('utf-8')

    text = TextBlob(comment)
    try:
        text = text.translate(from_lang='en', to=language)
        text = text.translate(from_lang=language, to='en')
    except Exception:
        pass

    return str(text)


def main():

    train_data = pd.read_csv('./inputs/train.csv')
    comments_list = train_data['comment_text'].fillna(NAN_WORD).values

    if not os.path.exists('./extended_data'):
        os.mkdir('./extended_data')

    parallel = Parallel(n_jobs=-1, backend='threading', verbose=5)
    for language in ['es', 'de', 'fr']:
        print(f'Translate comments using {language} language')

        translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
        train_data['comment_text'] = translated_data

        result_path = os.path.join('./extended_data/train_' + language + '.csv')
        train_data.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()