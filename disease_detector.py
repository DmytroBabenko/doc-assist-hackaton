from typing import List
import collections
import numpy as np
import json
import stanza
from transformers import pipeline
from scipy import spatial

# stanza.download(lang="ru")


def load_json(file: str):
    with open(file) as fp:
        data = json.load(fp)
        return data


class DiseaseDetector:
    ACCEPTABLE_UPOS = ['NOUN']
    THRESHOLD_SIMILARITY = 0.9
    MAX_NUM = 5

    def __init__(self, symptom_nouns_dict_file: str, body_nouns_dict_file: str,
                 symptom_diseases_dict_file: str, body_diseases_dict_file: str,
                 disease_id_to_name_dict_file: str):
        self.symptom_nouns_dict: dict = load_json(symptom_nouns_dict_file)
        self.body_nouns_dict: dict = load_json(body_nouns_dict_file)
        self.symptom_diseases_dict: dict = load_json(symptom_diseases_dict_file)
        self.body_diseases_dict: dict = load_json(body_diseases_dict_file)

        self.nlp_ru = stanza.Pipeline('ru')

        self.feature_extractor = pipeline("feature-extraction", model='bert-base-multilingual-uncased')

        self.disease_id_to_name_dict = load_json(disease_id_to_name_dict_file)

    @staticmethod
    def create_instance():
        disease_detector = DiseaseDetector(symptom_nouns_dict_file="./data/symptom_nouns_dict.json",
                                           body_nouns_dict_file="./data/body_nouns_dict.json",
                                           symptom_diseases_dict_file="./data/symptom_diseases_dict.json",
                                           body_diseases_dict_file="./data/body_diseases_dict.json",
                                           disease_id_to_name_dict_file="./data/disease_id_to_name_dict.json")

        return disease_detector

    def detect_disease(self, comment: str):
        comment_nouns = self.extract_nouns_from_text(comment)
        comment_noun_vectors = [self.get_feature_vector(noun) for noun in comment_nouns]

        symptom_match_id_dict = self._find_match(comment_nouns, comment_noun_vectors, self.symptom_nouns_dict)
        detected_diseases_by_symptoms = self.select_diseases_by_match_ids(symptom_match_id_dict,
                                                                          self.symptom_diseases_dict)

        body_match_id_dict = self._find_match(comment_nouns, comment_noun_vectors, self.body_nouns_dict)
        detected_diseases_by_bodies = self.select_diseases_by_match_ids(body_match_id_dict, self.body_diseases_dict)

        merged_diseases_dict = self._merge_two_detected_diseases(detected_diseases_by_symptoms,
                                                                 detected_diseases_by_bodies)


        possible_diseases = []
        for id, score in merged_diseases_dict.items():
            disease_id = str(id)
            if disease_id in self.disease_id_to_name_dict:
                possible_diseases.append(self.disease_id_to_name_dict[disease_id])

            if len(possible_diseases) >= self.MAX_NUM:
                break

        return possible_diseases

    def extract_nouns_from_text(self, text: str):
        nouns = []
        doc = self.nlp_ru(text)
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos not in self.ACCEPTABLE_UPOS:
                    continue

                nouns.append(word.lemma)
        return nouns

    def _find_match(self, comment_nouns: List[str], comment_noun_vectors, nouns_dict: dict):
        match_ids = collections.defaultdict(lambda: 0)
        size = len(comment_nouns)
        for i in range(size):
            comment_noun = comment_nouns[i]
            for id, symptom_item in nouns_dict.items():
                if comment_noun in symptom_item['nouns']:
                    match_ids[id] += 1
                else:
                    similarities = [DiseaseDetector.calc_similarity(comment_noun_vectors[i], v) for v in
                                      symptom_item['noun_vectors']]
                    if not similarities:
                        continue
                    similarity = max(similarities)
                    if similarity >= DiseaseDetector.THRESHOLD_SIMILARITY:
                        match_ids[id] += similarity

        return match_ids

    @staticmethod
    def select_diseases_by_match_ids(match_id_dict: dict, id_to_diseases: dict):
        selected_diseases_dict = collections.defaultdict(lambda: 0)
        for id, coefficient in match_id_dict.items():
            if id in id_to_diseases:
                for disease_id in id_to_diseases[id]:
                    selected_diseases_dict[disease_id] += coefficient

        return selected_diseases_dict

    def get_feature_vector(self, noun: str):
        vectors = self.feature_extractor(noun)
        return self.avg_vector(vectors[0])

    @staticmethod
    def calc_similarity(v1, v2):
        cosine = spatial.distance.cosine(v1, v2)
        return 1 - cosine

    @staticmethod
    def avg_vector(v):
        vector = np.zeros(768)
        for vi in v:
            vector += np.array(vi)

        return vector / len(v)

    @staticmethod
    def _merge_two_detected_diseases(detected_disease_dict1: dict, detected_disease_dict2: dict):
        merged_diseases_dict = collections.defaultdict(lambda: 0)
        ids = set(detected_disease_dict1.keys()).intersection(set(detected_disease_dict2.keys()))
        if not ids:
            for id in detected_disease_dict1:
                merged_diseases_dict[id] = detected_disease_dict1[id]
        else:
            for id in ids:
                merged_diseases_dict[id] = detected_disease_dict1[id] + detected_disease_dict2[id]

        merged_diseases_dict = {k: v for k, v in sorted(merged_diseases_dict.items(), key=lambda item: item[1], reverse=True)}
        return merged_diseases_dict


# disease_detector = DiseaseDetector.create_instance()
#
# diseases = disease_detector.detect_disease("Насморк, температура и головная боль")
#
# a = 10
