import re, string, unicodedata
import inflect
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


#Cleanning text
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    return [word.lower() for word in words]


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""

    # remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # return re.sub(remove_chars, '', words)

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    # words = remove_non_ascii(words)
    words = to_lowercase(words)         # 小写
    words = remove_punctuation(words)   # 去掉标点符号
    # remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # return re.sub(remove_chars, '', words)
    # words = replace_numbers(words)      # 数字替换为英文
    words = remove_stopwords(words)     # 去掉停用词
    return words


def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

stop_words = stopwords.words('english')
stop = set(stop_words)
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()         # 词干化


def process_sentencce_tokenizer(sentence: str):
    # stop_free = " ".join([i for i in sentence.lower().split() if i not in stop])
    # punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # normalized = [lemma.lemmatize(word) for word in punc_free.split()]

    sentence = sentence.lower()
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    sentence = re.sub(remove_chars, ' ', sentence)
    #
    words = nltk.word_tokenize(sentence)
    # words = [word for word in words if word not in stopwords.words('english')]
    # words = normalize(words)
    # stems, lemmas = stem_and_lemmatize(words)
    # lemmas = lemmatize_verbs(words)
    return words


if __name__ == "__main__":
    sample = 'I love u! \n How about u.\r'
    # For our task, we will tokenize our sample text into a list of words.#
    words = nltk.word_tokenize(sample)
    print(words)
    words = normalize(words)
    print(words)
    stems, lemmas = stem_and_lemmatize(words)
    print('Stemmed:\n', stems)
    print('\nLemmatized:\n', lemmas)
