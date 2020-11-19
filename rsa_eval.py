import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics

import sys

if '/home/anie/AutoGrade/simplepg' in sys.path:
    del sys.path[sys.path.index("/home/anie/AutoGrade/simplepg")]

from rsa import IncRSA
from rsa import load_model

puncs = set(string.punctuation)
en_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def load_inc_rsa_model(rsa_dataset):
    model = load_model(rsa_dataset)
    rsa_model = IncRSA(model, rsa_dataset)
    return rsa_model


class KeywordClassifier(object):
    def __init__(self, rsa_dataset):
        # we need:
        # 1. structure that is: {name: match_keywords}
        # 2. We go from organ -> aspect -> attributes
        # where we only do organ -> aspect (segment) first
        # For each segment, then we have:
        # {seg_name/id: [organ_name, aspect_name, attr_name]}

        self.rsa_dataset = rsa_dataset

        # organ first
        self.organ_name_to_match_words = {}
        # aspect second
        self.organ_name_to_aspect_name = {}
        self.attr_name_to_decomp = {}
        self.segment_name_to_decomp = {}

        for i, seg_attr in enumerate(self.rsa_dataset.attr_vocab_ls):
            a = seg_attr.split("::")
            b = a[0].split("_")
            organ_name = b[1]
            if len(b) == 2:
                aspect_name = None
            elif len(b) == 4:
                aspect_name = b[3]
                organ_name = b[1] + '_' + b[2]
            else:
                aspect_name = b[2]
            # aspect_name = b[2] if len(b) != 2 else None
            self.attr_name_to_decomp[seg_attr] = (organ_name, aspect_name, a[1])
            self.segment_name_to_decomp[seg_attr.split("::")[0]] = (organ_name, aspect_name)

            self.organ_name_to_match_words[organ_name] = {organ_name}

            # for bill-length
            if aspect_name == 'length':
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: {'long', 'short', 'longer', 'shorter'}}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = {'long', 'short', 'longer', 'shorter'}
                continue

            # we actually ignored "has_shape" and "has_size"
            if aspect_name is not None:
                attr_set = self.get_attr_descriptor_for_organ(organ_name, aspect_name)
                if organ_name not in self.organ_name_to_aspect_name:
                    self.organ_name_to_aspect_name[organ_name] = {aspect_name: attr_set}
                else:
                    self.organ_name_to_aspect_name[organ_name][aspect_name] = attr_set

                # update pattern keywords!
                if organ_name != 'head' and aspect_name == 'pattern':
                    # add pattern descriptors here
                    # every organ beside head shares same pattern descriptor
                    self.fill_organ_aspect_key_words(organ_name, aspect_name, ['striped', 'stripe',
                                                                               'speckle', 'speckled',
                                                                               'multicolored', 'multicolor',
                                                                               'specks', 'speck',
                                                                               'ornate', 'scattered',
                                                                               'coloring', 'spots', 'spot',
                                                                               'rounded', 'mottled',
                                                                               'tuft', 'webbed', 'puffy',
                                                                               'pointy'])

        # for organ names, most are FINE
        # but not for "upperparts", "underparts", "back", "under_tail", "size", "shape", "primary", 'upper_tail'
        self.expand()
        self.aspect_expand()

        for name in ['wing', 'throat', 'head', 'forehead', 'nape', 'upper_tail', 'crown', 'breast']:
            self.organ_name_to_match_words['upperparts'].update(self.organ_name_to_match_words[name])

        for name in ['leg', 'under_tail', 'belly']:
            self.organ_name_to_match_words['underparts'].update(self.organ_name_to_match_words[name])

        self.color_match_words = self.get_attr_descriptors("color")
        self.color_match_words.update(['navy', 'bluish', 'violet', 'scarlet', 'greenish', 'silrumpver', 'teal',
                                       'pinkish', 'colored', 'color', 'multicolored', 'multicolor',
                                       'tan', 'bright', 'dark', 'brown', 'brownish', 'vibrant',
                                       'gray', 'pale', 'russet', 'yellow', 'orange', 'golden',
                                       'coloring', 'toned', 'shiny', 'pink', 'vivid', 'blackish'])
        self.fill_color_key_words()

        self.organ_name_to_match_words['size'] = ['large', 'small', 'very large', 'medium', 'very small',
                                                  'petite', 'pudgy', 'smal', 'slim', 'huge', 'elongated',
                                                  'skinny', 'sized', 'thick', 'short', 'long', 'shorter', 'longer',
                                                  'puffy']
        self.organ_name_to_match_words['shape'] = ['plump', 'mohawk', 'perching', 'perch', 'gull', 'humming',
                                                   'clinging', 'hawk', 'rounded', 'round', 'puffy']

    def print_descriptors(self, skip_color=True):
        for organ_name, aspect_name_match_words in self.organ_name_to_aspect_name.items():
            if 'color' in aspect_name_match_words:
                if skip_color:
                    continue
            print(organ_name, aspect_name_match_words)
            print()

    def get_attr_descriptors(self, aspect_word):
        color_attrs = [t.split("::")[0] for t in self.rsa_dataset.q_id_to_segments if aspect_word in t]
        uniq_colors = set()
        for c_a in color_attrs:
            colors = [t.split("::")[1] for t in self.rsa_dataset.attr_vocab_ls if c_a in t]
            uniq_colors.update(colors)

        return uniq_colors

    def get_attr_descriptor_for_organ(self, organ_name, aspect_word, verbose=False):
        attrs = [t for t in self.rsa_dataset.attr_vocab_ls if organ_name in t and aspect_word in t]
        if verbose:
            print(attrs)
        uniq_attrs = set()
        for a in attrs:
            a = a.split("::")[1]
            if a == 'curved_(up_or_down)':
                uniq_attrs.add('curved')
                uniq_attrs.add('up')
                uniq_attrs.add('down')
            elif '_' in a:
                uniq_attrs.update(a.split('_'))
            elif 'wings' in a:
                a = a.split('-')[0]
                uniq_attrs.add(a)
            else:
                uniq_attrs.add(a)

        return uniq_attrs

    def aspect_expand(self):
        self.fill_organ_aspect_key_words('bill', 'shape', ['triangular', 'pointed', 'curved', 'pointy'])
        self.fill_organ_aspect_key_words('bill', 'length', ['large', 'small', 'tiny', 'huge'])
        self.fill_organ_aspect_key_words('tail', 'shape', ['fan'])
        self.fill_organ_aspect_key_words('head', 'pattern', ['streak'])
        self.fill_organ_aspect_key_words('wing', 'shape', ['long', 'large'])

    def fill_organ_aspect_key_words(self, organ_name, aspect_name, keywords):
        self.organ_name_to_aspect_name[organ_name][aspect_name].update(keywords)

    def expand(self):
        """
        {'bill': {'bill'},
         'wing': {'wing'},
         'upperparts': {'upperparts'},
         'underparts': {'underparts'},
         'breast': {'breast'},
         'back': {'back'},
         'tail': {'tail'},
         'upper_tail': {'upper_tail'},
         'head': {'head'},
         'throat': {'throat'},
         'eye': {'eye'},
         'forehead': {'forehead'},
         'under_tail': {'under_tail'},
         'nape': {'nape'},
         'belly': {'belly'},
         'size': {'size'},
         'shape': {'shape'},
         'primary': {'primary'},
         'leg': {'leg'},
         'crown': {'crown'}}
        """
        # we lemmatize everything, so it's fine-
        self.fill_organ_key_words("leg", ['tarsal', 'tarsals', 'tarsuses', 'tarsus', 'foot', 'thighs',
                                          'feet', 'claws', 'claw', 'legs'])
        self.fill_organ_key_words("wing", ['wingbars', 'wingbar', 'rectricles', 'rectricle', 'retrice',
                                           'gull', 'tip', 'tips', 'primaries', 'primary', 'secondaries',
                                           'secondary', 'converts', 'convert', 'retrices',
                                           'wingspan', 'wingspans'])
        self.fill_organ_key_words('head', ['malar', 'malars', 'cheekpatch', 'eyebrows', 'cheek',
                                           'superciliary', 'eyebrow', 'eyering', 'eyeline', 'eyelines',
                                           'eyerings', 'ring', 'rings'])
        self.fill_organ_key_words('bill', ['beek', 'beaks', 'beak', 'beeks', 'hook', 'bil'])
        self.fill_organ_key_words('under_tail', ['undertail', 'tail', 'rump'])
        self.fill_organ_key_words('belly', ['underbelly', 'stomach', 'plumage', 'feather', 'feathers',
                                            'abdomen', 'side'])
        self.fill_organ_key_words('breast', ['chest', 'stomach', 'plumage', 'feather', 'feathers', 'breasted'])
        self.fill_organ_key_words('upperparts', ['body', 'side', 'sides'])
        self.fill_organ_key_words('forehead', ['forehead'])  # 'eyebrows', 'eyebrow', 'eyering'
        self.fill_organ_key_words('back', ['plumage'])
        self.fill_organ_key_words('primary', ['primaries'])
        self.fill_organ_key_words('throat', ['neck'])
        self.fill_organ_key_words('crown', ['crest'])
        self.fill_organ_key_words('upper_tail', ['rump', 'tail'])
        self.fill_organ_key_words('eye', ['eyebrows', 'superciliary', 'eyebrow', 'eyering', 'eyeline', 'eyelines',
                                          'eyerings', 'ring', 'rings'])

    def fill_organ_key_words(self, name, list_of_words):
        # we collect a lot of them and fill them up
        self.organ_name_to_match_words[name].update(list_of_words)

    def fill_color_key_words(self):
        # iterate through ALL
        for organ_name, aspect_name_match_words in self.organ_name_to_aspect_name.items():
            if 'color' in aspect_name_match_words:
                aspect_name_match_words['color'].update(self.color_match_words)

    def classify_parts(self, part_name, text, tokenize=False):
        # basically we just try different ways to change the text and match
        assert part_name in self.organ_name_to_match_words

        keywords = self.organ_name_to_match_words[part_name]

        words = nltk.word_tokenize(text) if tokenize else text

        found, ind_list = False, []

        # then we lemmatize text
        for i, w in enumerate(words):
            if w in keywords:
                found = True
                ind_list.append(i)
                continue
            w = lemmatizer.lemmatize(w.lower())
            if w in keywords:
                found = True
                ind_list.append(i)

        return found, ind_list

    def classify_parts_aspect(self, part_name, aspect_name, text, window_size=3, tokenize=False):
        # if no tokenize, we expect a list of words
        assert part_name in self.organ_name_to_match_words
        assert aspect_name in self.organ_name_to_aspect_name[part_name]

        # here we first identify the location of body parts
        # then we look ahead for a fixed window (3-5 words) for the aspect match
        if tokenize:
            text = nltk.word_tokenize(text)

        assert type(text) == list
        found, idx_list = self.classify_parts(part_name, text, tokenize=False)

        keywords = self.organ_name_to_aspect_name[part_name][aspect_name]

        if not found:
            return False
        else:
            for i in idx_list:
                # check previous 5 words
                # a = [0,1,2,3,4]
                # a[1:3] = [1, 2]
                lookahead_idx = max(i - window_size, 0)
                text_span = text[lookahead_idx:i]
                for t in text_span:
                    if t in keywords:
                        return True

        return False


class KeywordExtractor(object):
    """
    Actually, this is quite powerful
    """

    def __init__(self, rsa_dataset):
        self.rsa_dataset = rsa_dataset
        self.X_words, self.img_ids = self.build_dataset()
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))

        self.X = self.vectorizer.fit_transform(self.X_words)
        self.vocab = self.vectorizer.get_feature_names()

    def build_dataset(self):
        X_words = []
        img_ids = []
        for img_id, split in self.rsa_dataset.image_id_to_split.items():
            img_ids.append(img_id)
            label = [0.] * len(self.rsa_dataset.q_id_to_segments)

            caption_list = self.rsa_dataset.get_caption_by_img_id(img_id, True)
            # we concat 5 captions into ONE
            total_caption = " </s> ".join(caption_list)
            X_words.append(total_caption)

            # get y-label
            img_pos = self.rsa_dataset.filename_to_cub_img_id[img_id]
            random_img_pos = self.rsa_dataset.img_id_to_random_idx[img_pos]

            attr_vec = self.rsa_dataset.attribute_matrix[random_img_pos, :]
            for seg_label, attr_ids in self.rsa_dataset.segments_to_attr_id.items():
                if sum(attr_vec[attr_ids]) >= 1:
                    label_idx = self.rsa_dataset.q_id_to_segments.index(seg_label)
                    label[label_idx] = 1.

        return X_words, img_ids

    def print_attr_for_key(self, keyword):
        focus_att_ids = [i for i in range(len(self.rsa_dataset.attr_vocab_ls)) if
                         keyword in self.rsa_dataset.attr_vocab_ls[i]]
        for att_id in focus_att_ids:
            print(self.rsa_dataset.attr_vocab_ls[att_id])

    def collect_captions_for_key(self, key_word):
        # this would partition a dataset given a keyword
        # we will get indices over attributes that contain that keywords
        # then divide the dataset in half for it
        # get unigram, and then do it
        # we return a Y-label for it

        focus_att_ids = [i for i in range(len(self.rsa_dataset.attr_vocab_ls)) if
                         key_word in self.rsa_dataset.attr_vocab_ls[i]]

        y = []
        # we iterate through the img_id!
        for img_id in self.img_ids:
            # get y-label
            img_pos = self.rsa_dataset.filename_to_cub_img_id[img_id]
            random_img_pos = self.rsa_dataset.img_id_to_random_idx[img_pos]
            attr_vec = self.rsa_dataset.attribute_matrix[random_img_pos, :]

            if sum(attr_vec[focus_att_ids]) >= 1:
                y.append(1)
            else:
                y.append(0)

        return np.array(y)

    def get_top_keywords_for(self, attr_word, top=20):
        # we compose a dataset, train a lasso over unigram
        # hopefully this all is fast enough...otherwise we'll just do some tf-idf stuff
        y = self.collect_captions_for_key(attr_word)
        print("positive class percentage: {}".format(y.mean()))
        if y.mean() > 0.9:
            print("Warning: too imbalanced, do not use")

        reg = linear_model.LogisticRegression(fit_intercept=True, penalty='l1', solver='saga',
                                              max_iter=100)
        reg.fit(self.X, y)
        y_hat = reg.predict(self.X)
        print(metrics.classification_report(y, y_hat, digits=3))
        print("Accuracy: ", metrics.accuracy_score(y, y_hat))

        _, coords = np.nonzero(reg.coef_)
        print("Number of non-zero words: {}".format(len(coords)))
        word_coeff = []
        for c in coords:
            word_coeff.append((self.vocab[c], reg.coef_[0, c]))

        word_coeff = sorted(word_coeff, key=lambda x: x[1], reverse=True)[:top]

        return word_coeff
