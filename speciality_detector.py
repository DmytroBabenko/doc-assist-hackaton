import json
import pickle

import stanza
from sklearn.feature_extraction.text import CountVectorizer


def load_json(file: str):
    with open(file) as fp:
        data = json.load(fp)
        return data

def dummy(doc):
    return doc


class SpecialtyDetector:
    ACCEPTABLE_UPOS = ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']

    def __init__(self):
        self.nlp_ru = stanza.Pipeline('ru')
        self.stopwords_ru = load_json('stopwords-ru.json')
        self.cv = CountVectorizer(decode_error="replace", tokenizer=dummy, preprocessor=dummy,
                                  vocabulary=pickle.load(open("./data/count_vectorizer_vocab.pkl", "rb")))

        self.model = pickle.load(open("./data/multinomialNB.sav", 'rb'))
        self.specialty_to_group = load_json("./data/group_specialty_dict.json")
        self.group_id_to_speciality_id = {int(v): int(k) for k, v in self.specialty_to_group.items()}
        self.speciality_id_name = load_json("./data/speciality_id_name_dict.json")

    def detect(self, text: str) -> str:
        try:
        # if True:
            features = self.extract_feature(text)
            y = self.model.predict(features)[0]
            speciality_id = self.group_id_to_speciality_id[y]
            name = self.speciality_id_name[str(speciality_id)]

            return name
        except:
            return "Unknown"

    def extract_feature(self, text: str):
        tokens = self.get_important_tokens(text)
        x = self.cv.transform([tokens])
        return x

    def get_important_tokens(self, text: str):
        tokens = []
        try:
            sentences = self.nlp_ru(text).sentences
            for sent in sentences:
                for word in sent.words:
                    lemma = word.lemma
                    if lemma in self.stopwords_ru:
                        continue
                    if word.upos not in self.ACCEPTABLE_UPOS:
                        continue
                    tokens.append(lemma)
        except:
            pass
        return tokens

#
# detector = SpecialtyDetector()
#
#
# speciality_name = detector.detect("Насморк, горло")
# print(speciality_name)


