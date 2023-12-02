# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sqlite3
from logging import getLogger
from string import punctuation
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import nltk
import pymorphy2
from hdt import HDTDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from models.entity_extraction.find_word import WordSearcher

log = getLogger(__name__)
nltk.download("stopwords")
nltk.download('omw-1.4')


@register("entity_linker")
class EntityLinker(Component, Serializable):
    """
    Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(
            self,
            load_path: str,
            entity_ranker=None,
            entities_database_filename: str = None,
            words_dict_filename: str = None,
            ngrams_matrix_filename: str = None,
            num_entities_for_bert_ranking: int = 50,
            num_entities_for_conn_ranking: int = 5,
            num_entities_to_return: int = 10,
            max_text_len: int = 300,
            max_paragraph_len: int = 150,
            lang: str = "en",
            use_descriptions: bool = True,
            use_tags: bool = False,
            lemmatize: bool = False,
            fuzzy_match: bool = False,
            full_paragraph: bool = False,
            use_connections: bool = False,
            sort_low_conf: bool = False,
            kb_filename: str = None,
            prefixes: Dict[str, Any] = None,
            **kwargs,
    ) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            entity_ranker: component deeppavlov.models.kbqa.rel_ranking_bert
            num_entities_for_bert_ranking: number of candidate entities for BERT ranking using description and context
            ngram_range: char ngrams range for TfidfVectorizer
            num_entities_to_return: number of candidate entities for the substring which are returned
            lang: russian or english
            use_description: whether to perform entity ranking by context and description
            lemmatize: whether to lemmatize tokens
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.lemmatize = lemmatize
        self.fuzzy_match = fuzzy_match
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.num_entities_for_conn_ranking = num_entities_for_conn_ranking
        self.entity_ranker = entity_ranker
        self.entities_database_filename = entities_database_filename
        self.words_dict_filename = words_dict_filename
        self.ngrams_matrix_filename = ngrams_matrix_filename
        self.num_entities_to_return = num_entities_to_return
        self.max_text_len = max_text_len
        self.max_paragraph_len = max_paragraph_len
        self.lang = f"@{lang}"
        if self.lang == "@en":
            self.stopwords = set(stopwords.words("english"))
            nltk.download('wordnet')
            self.morph = WordNetLemmatizer()
        elif self.lang == "@ru":
            self.stopwords = set(stopwords.words("russian"))
            self.morph = pymorphy2.MorphAnalyzer()
        self.use_descriptions = use_descriptions
        self.use_connections = use_connections
        self.sort_low_conf = sort_low_conf
        self.use_tags = use_tags
        self.full_paragraph = full_paragraph
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.not_found_str = "not in wiki"
        self.related_tags = {
            "loc": ["gpe", "country", "city", "us_state", "river"],
            "gpe": ["loc", "country", "city", "us_state"],
            "work_of_art": ["product", "law"],
            "product": ["work_of_art"],
            "law": ["work_of_art"],
            "org": ["fac", "business"],
            "business": ["org"]
        }
        self.word_searcher = None
        if self.words_dict_filename:
            self.word_searcher = WordSearcher(self.words_dict_filename, self.ngrams_matrix_filename, self.lang)
        self.kb_filename = kb_filename
        self.prefixes = prefixes
        self.load()

    def load(self) -> None:
        self.conn = sqlite3.connect(str(self.load_path / self.entities_database_filename))
        self.cur = self.conn.cursor()
        self.kb = None
        if self.kb_filename:
            self.kb = HDTDocument(str(expand_path(self.kb_filename)))

    def save(self) -> None:
        pass

    def __call__(
            self,
            substr_batch: List[List[str]],
            tags_batch: List[List[str]] = None,
            probas_batch: List[List[float]] = None,
            sentences_batch: List[List[str]] = None,
            offsets_batch: List[List[List[int]]] = None,
            sentences_offsets_batch: List[List[Tuple[int, int]]] = None,
            entities_to_link_batch: List[List[int]] = None
    ):
        if (not sentences_offsets_batch or sentences_offsets_batch[0] is None) and sentences_batch is not None:
            sentences_offsets_batch = []
            for sentences_list in sentences_batch:
                sentences_offsets_list = []
                start = 0
                for sentence in sentences_list:
                    end = start + len(sentence)
                    sentences_offsets_list.append([start, end])
                    start = end + 1
                sentences_offsets_batch.append(sentences_offsets_list)

        if sentences_batch is None:
            sentences_batch = [[] for _ in substr_batch]
            sentences_offsets_batch = [[] for _ in substr_batch]

        if not entities_to_link_batch or entities_to_link_batch[0] is None:
            entities_to_link_batch = [[1 for _ in substr_list] for substr_list in substr_batch]

        log.debug(f"substr: {substr_batch} --- sentences_batch: {sentences_batch} --- offsets: {offsets_batch}")
        if (not offsets_batch or offsets_batch[0] is None) and sentences_batch:
            offsets_batch = []
            for substr_list, sentences_list in zip(substr_batch, sentences_batch):
                text = " ".join(sentences_list).lower()
                log.debug(f"text {text}")
                offsets_list = []
                for substr in substr_list:
                    st_offset = text.find(substr.lower())
                    end_offset = st_offset + len(substr)
                    offsets_list.append([st_offset, end_offset])
                offsets_batch.append(offsets_list)
        ids_batch, conf_batch, pages_batch, labels_batch = [], [], [], []
        for substr_list, offsets_list, tags_list, probas_list, sentences_list, sentences_offsets_list, \
                entities_to_link in zip(substr_batch, offsets_batch, tags_batch, probas_batch, sentences_batch,
                                        sentences_offsets_batch, entities_to_link_batch):
            ids_list, conf_list, pages_list, labels_list = \
                self.link_entities(substr_list, offsets_list, tags_list, probas_list, sentences_list,
                                   sentences_offsets_list, entities_to_link)
            log.debug(f"ids_list {ids_list} conf_list {conf_list}")
            if self.num_entities_to_return == 1:
                pages_list = [pages[0] for pages in pages_list]
            else:
                pages_list = [pages[: len(ids)] for pages, ids in zip(pages_list, ids_list)]
            ids_batch.append(ids_list)
            conf_batch.append(conf_list)
            pages_batch.append(pages_list)
            labels_batch.append(labels_list)
        return ids_batch, conf_batch, pages_batch, labels_batch

    def link_entities(
            self,
            substr_list: List[str],
            offsets_list: List[List[int]],
            tags_list: List[str],
            probas_list: List[float],
            sentences_list: List[str],
            sentences_offsets_list: List[List[int]],
            entities_to_link: List[int]
    ) -> List[List[str]]:
        log.debug(f"substr_list {substr_list} tags_list {tags_list} probas {probas_list} offsets_list {offsets_list}")
        ids_list, conf_list, pages_list, label_list, descr_list = [], [], [], [], []
        if substr_list:
            entities_scores_list = []
            cand_ent_scores_list = []
            for substr, tags, proba in zip(substr_list, tags_list, probas_list):
                for symb in punctuation:
                    substr = substr.replace(symb, " ")
                substr = re.sub(r'\s+', ' ', substr)
                substr = substr.strip()
                cand_ent_init = defaultdict(set)
                if len(substr) > 1:
                    if isinstance(tags, str):
                        tags = [tags]
                    tags = [tag.lower() for tag in tags]
                    if tags and not isinstance(tags[0], (list, tuple)):
                        tags = [(tag, 1.0) for tag in tags]
                    if tags and tags[0][0] == "e":
                        use_tags_flag = False
                    else:
                        use_tags_flag = True
                    cand_ent_init = self.find_exact_match(substr, tags, use_tags=use_tags_flag)
                    new_substr = re.sub(r"\b([a-z]{1}) ([a-z]{1})\b", r"\1\2", substr)
                    if substr != new_substr:
                        new_cand_ent_init = self.find_exact_match(new_substr, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    init_substr_split = substr.lower().split(" ")
                    if tags[0][0] in {"person", "work_of_art"}:
                        substr_split = [word for word in substr.lower().split(" ") if len(word) > 0]
                    else:
                        substr_split = [word for word in substr.lower().split(" ")
                                        if word not in self.stopwords and len(word) > 0]
                    substr_lemm1 = ""
                    substr_split_lemm1 = []
                    substr_split_lemm1, substr_split_lemm2 = self.morph_parse(substr_split)
                    substr_lemm1 = " ".join(substr_split_lemm1)
                    substr_lemm2 = " ".join(substr_split_lemm2)
                    if substr_split != substr_split_lemm1 \
                            or (tags[0][0] == "work_of_art"
                                and len(substr_split) != len(init_substr_split)):
                        new_cand_ent_init = self.find_fuzzy_match(substr_split, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                    if substr_split != substr_split_lemm1:
                        new_cand_ent_init = self.find_exact_match(substr_lemm1, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                        new_cand_ent_init = self.find_fuzzy_match(substr_split_lemm1, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                    if substr_split != substr_split_lemm2 and substr_split_lemm1 != substr_split_lemm2:
                        new_cand_ent_init = self.find_exact_match(substr_lemm2, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                        new_cand_ent_init = self.find_fuzzy_match(substr_split_lemm2, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    all_low_conf = self.define_all_low_conf(cand_ent_init, 1.0)
                    clean_tags, corr_tags, corr_clean_tags = self.correct_tags(tags)
                    log.debug(f"substr: {substr} --- lemm: {substr_split_lemm1} --- tags: {tags} --- corr_tags: "
                              f"{corr_tags} --- all_low_conf: {all_low_conf} --- cand_ent_init: {len(cand_ent_init)}")

                    if (not cand_ent_init or all_low_conf) and corr_tags:
                        corr_cand_ent_init = self.find_exact_match(substr, corr_tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, corr_cand_ent_init)
                        if substr_split != substr_split_lemm1:
                            new_cand_ent_init = self.find_exact_match(substr_lemm1, corr_tags, use_tags=use_tags_flag)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                            new_cand_ent_init = self.find_fuzzy_match(substr_split_lemm1, corr_tags,
                                                                      use_tags=use_tags_flag)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    if not cand_ent_init and len(substr_split) == 1 and self.word_searcher:
                        corr_words = self.word_searcher(substr_split[0], set(clean_tags + corr_clean_tags))
                        if corr_words:
                            cand_ent_init = self.find_exact_match(corr_words[0], tags + corr_tags,
                                                                  use_tags=use_tags_flag)

                    if (not cand_ent_init or self.fuzzy_match) and len(substr_split) > 1:
                        cand_ent_init = self.find_fuzzy_match(substr_split, tags)

                    all_low_conf = self.define_all_low_conf(cand_ent_init, 0.85)
                    if (not cand_ent_init or all_low_conf) and tags[0][0] != "t":
                        use_tags_flag = False
                        new_cand_ent_init = self.find_exact_match(substr, tags, use_tags=use_tags_flag)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                        if substr_split != substr_split_lemm1 and (tags[0][0] == "e" or not cand_ent_init):
                            new_cand_ent_init = self.find_fuzzy_match(substr_split, tags, use_tags=use_tags_flag)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                            new_cand_ent_init = self.find_fuzzy_match(substr_split_lemm1, tags, use_tags=use_tags_flag)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                cand_ent_scores = []
                for entity in cand_ent_init:
                    entities_scores = list(cand_ent_init[entity])
                    entities_scores = sorted(entities_scores, key=lambda x: (x[0], x[2], x[1]), reverse=True)
                    cand_ent_scores.append(([entity] + list(entities_scores[0])))

                cand_ent_scores = sorted(cand_ent_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
                cand_ent_scores = cand_ent_scores[: self.num_entities_for_bert_ranking]
                cand_ent_scores_list.append(cand_ent_scores)
                entity_ids = [elem[0] for elem in cand_ent_scores]
                scores = [elem[1:4] for elem in cand_ent_scores]
                entities_scores_list.append(
                    {entity_id: entity_scores for entity_id, entity_scores in zip(entity_ids, scores)}
                )
                ids_list.append(entity_ids)
                conf_list.append(scores)
                pages = [elem[4] for elem in cand_ent_scores]
                entity_labels = [elem[5] for elem in cand_ent_scores]
                pages_list.append({entity_id: page for entity_id, page in zip(entity_ids, pages)})
                label_list.append(
                    {entity_id: entity_label for entity_id, entity_label in zip(entity_ids, entity_labels)})
                descr_list.append([elem[6] for elem in cand_ent_scores])

            scores_dict = {}
            if self.use_connections and self.kb:
                scores_dict = self.rank_by_connections(ids_list)

            if self.use_descriptions:
                substr_lens = [len(entity_substr.split()) for entity_substr in substr_list]
                ids_list, conf_list = self.rank_by_description(substr_list, tags_list, offsets_list, ids_list,
                                                               descr_list, entities_scores_list, sentences_list,
                                                               sentences_offsets_list, substr_lens, scores_dict)
        label_list = [[label_dict.get(entity_id, "") for entity_id in entity_ids]
                      for entity_ids, label_dict in zip(ids_list, label_list)]
        pages_list = [[pages_dict.get(entity_id, "") for entity_id in entity_ids]
                      for entity_ids, pages_dict in zip(ids_list, pages_list)]

        f_ids_list, f_conf_list, f_pages_list, f_label_list = [], [], [], []
        for ids, confs, pages, labels, add_flag in \
                zip(ids_list, conf_list, pages_list, label_list, entities_to_link):
            if add_flag:
                f_ids_list.append(ids)
                f_conf_list.append(confs)
                f_pages_list.append(pages)
                f_label_list.append(labels)
        return f_ids_list, f_conf_list, f_pages_list, f_label_list

    def define_all_low_conf(self, cand_ent_init, thres):
        all_low_conf = True
        for entity_id in cand_ent_init:
            entity_info_set = cand_ent_init[entity_id]
            for entity_info in entity_info_set:
                if entity_info[0] >= thres:
                    all_low_conf = False
                    break
            if not all_low_conf:
                break
        return all_low_conf

    def correct_tags(self, tags):
        clean_tags = [tag for tag, conf in tags]
        corr_tags, corr_clean_tags = [], []
        for tag, conf in tags:
            if tag in self.related_tags:
                corr_tag_list = self.related_tags[tag]
                for corr_tag in corr_tag_list:
                    if corr_tag not in clean_tags and corr_tag not in corr_clean_tags:
                        corr_tags.append([corr_tag, conf])
                        corr_clean_tags.append(corr_tag)
        return clean_tags, corr_tags, corr_clean_tags

    def unite_dicts(self, cand_ent_init, new_cand_ent_init):
        for entity_id in new_cand_ent_init:
            if entity_id in cand_ent_init:
                for entity_info in new_cand_ent_init[entity_id]:
                    cand_ent_init[entity_id].add(entity_info)
            else:
                cand_ent_init[entity_id] = new_cand_ent_init[entity_id]
        return cand_ent_init

    def nomn_case(self, word):
        morph_parse_tok = self.morph.parse(word)[0].inflect({"nomn"})
        if morph_parse_tok:
            normal_form = morph_parse_tok.word
        else:
            normal_form = word
        return normal_form

    def normal_form(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        if morph_parse_tok:
            normal_form = morph_parse_tok.normal_form
        else:
            normal_form = word
        return normal_form

    def morph_parse(self, words):
        if self.lang == "@ru":
            words = [self.nomn_case(word) for word in words]
            if len(words) > 1 and self.morph.parse(words[-1])[0].tag.POS == "NOUN":
                gender = self.morph.parse(words[-1])[0].tag.gender
                if gender:
                    for i in range(len(words) - 1):
                        if self.morph.parse(words[i])[0].tag.POS == "ADJF":
                            words[i] = self.morph.parse(words[i])[0].inflect({gender}).word
            words_n = [self.normal_form(word) for word in words]
        elif self.lang == "@en":
            words = [self.morph.lemmatize(word) for word in words]
            words_n = words
        return words, words_n

    def process_cand_ent(self, cand_ent_init, entities_and_ids, substr_split, tag, tag_conf, use_tags):
        for title, entity_id, rels, ent_tag, page, label, descr in entities_and_ids:
            if (ent_tag == tag and use_tags) or not use_tags:
                substr_score = self.calc_substr_score(title, substr_split, entity_id, tag, ent_tag, label)
                cand_ent_init[entity_id].add((substr_score, rels, tag_conf, page, label, descr))
        return cand_ent_init

    def sanitize_substr(self, entity_substr, tag):
        if tag == "person":
            entity_substr_split = entity_substr.split()
            if len(entity_substr_split) > 1 and len(entity_substr_split[-1]) > 1 and len(entity_substr_split[-2]) == 1:
                entity_substr = entity_substr_split[-1]
        return entity_substr

    def find_exact_match(self, entity_substr, tags, use_tags=True):
        entity_substr = entity_substr.lower()
        entity_substr_split = entity_substr.split()
        cand_ent_init = defaultdict(set)
        for tag, tag_conf in tags:
            entity_substr = self.sanitize_substr(entity_substr, tag)
            query = "SELECT * FROM inverted_index WHERE title='{}';".format(entity_substr.lower())
            entities_and_ids = []
            try:
                res = self.cur.execute(query)
                entities_and_ids = res.fetchall()
            except:
                log.info(f"error in query execute {query}")
            if entities_and_ids:
                cand_ent_init = self.process_cand_ent(
                    cand_ent_init, entities_and_ids, entity_substr_split, tag, tag_conf, use_tags)
        return cand_ent_init

    def find_fuzzy_match(self, entity_substr_split, tags, use_tags=True):
        cand_ent_init = defaultdict(set)
        for tag, tag_conf in tags:
            for word in entity_substr_split:
                if len(word) > 1 and word not in self.stopwords:
                    query = "SELECT * FROM inverted_index WHERE title='{}';".format(word.lower())
                    part_entities_and_ids = []
                    try:
                        res = self.cur.execute(query)
                        part_entities_and_ids = res.fetchall()
                    except:
                        log.info(f"error in query execute {query}")
                    if part_entities_and_ids:
                        cand_ent_init = self.process_cand_ent(
                            cand_ent_init, part_entities_and_ids, entity_substr_split, tag, tag_conf, use_tags)
        return cand_ent_init

    def match_tokens(self, entity_substr_split, label_tokens):
        cnt = 0.0
        if not (len(entity_substr_split) > 1 and len(label_tokens) > 1 \
                and set(entity_substr_split) != set(label_tokens) and label_tokens[0] != label_tokens[-1] \
                and ((entity_substr_split[0] == label_tokens[-1]) or (entity_substr_split[-1] == label_tokens[0]))):
            for ent_tok in entity_substr_split:
                found = False
                for label_tok in label_tokens:
                    if label_tok == ent_tok:
                        found = True
                        break
                if found:
                    cnt += 1.0
                else:
                    for label_tok in label_tokens:
                        if label_tok[:2] == ent_tok[:2]:
                            fuzz_score = fuzz.ratio(label_tok, ent_tok)
                            if (fuzz_score >= 75.0 \
                                    or (len(label_tok) >= 8 and label_tok[:6] == ent_tok[:6] and fuzz_score > 70.0) \
                                    or (len(label_tokens) > 2 and len(label_tok) > 3 and label_tok[:4] == ent_tok[:4])) \
                                    and not found:
                                cnt += fuzz_score * 0.01
                                break
        substr_score = round(cnt / max(len(label_tokens), len(entity_substr_split)), 3)
        if len(label_tokens) == 2 and len(entity_substr_split) == 1:
            if entity_substr_split[0] == label_tokens[1]:
                substr_score = 0.5
            elif entity_substr_split[0] == label_tokens[0]:
                substr_score = 0.3
        return substr_score

    def correct_substr_score(self, entity_substr_split, label_tokens, substr_score):
        if sum([len(tok) == 1 for tok in entity_substr_split]) == 2 and len(label_tokens) >= 2 \
                and any([(len(tok) == 2 and re.findall(r"[a-z]{2}", tok)) for tok in label_tokens]):
            new_label_tokens = []
            for tok in label_tokens:
                if len(tok) == 2 and re.findall(r"[a-z]{2}", tok):
                    new_label_tokens.append(tok[0])
                    new_label_tokens.append(tok[1])
                else:
                    new_label_tokens.append(tok)
            label_tokens = new_label_tokens
        if any([re.findall(r"[\d]{4}", tok) for tok in entity_substr_split]) \
                and any([re.findall(r"[\d]{4}–[\d]{2}", tok) for tok in label_tokens]):
            new_label_tokens = []
            for tok in label_tokens:
                if re.findall(r"[\d]{4}–[\d]{2}", tok):
                    new_label_tokens.append(tok[:4])
                    new_label_tokens.append(tok[5:])
                else:
                    new_label_tokens.append(tok)
            label_tokens = new_label_tokens
        new_substr_score = self.match_tokens(entity_substr_split, label_tokens)
        substr_score = max(substr_score, new_substr_score)
        return substr_score

    def calc_substr_score(self, entity_title, entity_substr_split, entity_id, tag, ent_tag, entity_label):
        if self.lang == "@ru":
            entity_title = entity_title.replace("ё", "е")
        label_tokens = entity_title.split()
        substr_score = self.match_tokens(entity_substr_split, label_tokens)
        substr_score = self.correct_substr_score(entity_substr_split, label_tokens, substr_score)
        if tag == ent_tag and tag.lower() == "person" and len(entity_substr_split) > 1 \
                and len(entity_substr_split[-1]) > 1 and len(entity_substr_split[-2]) == 1 \
                and len(label_tokens) == len(entity_substr_split):
            cnt = 0.0
            for j in range(len(label_tokens) - 1):
                if label_tokens[j][0] == entity_substr_split[j][0]:
                    cnt += 1.0
            if label_tokens[-1] == entity_substr_split[-1]:
                cnt += 1.0
            new_substr_score = cnt / len(label_tokens)
            substr_score = max(substr_score, new_substr_score)

        if entity_title.lower() == entity_label.lower() and substr_score == 1.0:
            substr_score = substr_score * 1.1
        return substr_score

    def rank_by_description(
            self,
            entity_substr_list: List[str],
            tags_list: List[str],
            entity_offsets_list: List[List[int]],
            cand_ent_list: List[List[str]],
            cand_ent_descr_list: List[List[str]],
            entities_scores_list: List[Dict[str, Tuple[int, float]]],
            sentences_list: List[str],
            sentences_offsets_list: List[Tuple[int, int]],
            substr_lens: List[int],
            scores_dict: Dict[str, int] = None
    ) -> List[List[str]]:
        entity_ids_list = []
        conf_list = []
        contexts = []
        for entity_start_offset, entity_end_offset in entity_offsets_list:
            sentence = ""
            rel_start_offset = 0
            rel_end_offset = 0
            found_sentence_num = 0
            for num, (sent, (sent_start_offset, sent_end_offset)) in enumerate(
                    zip(sentences_list, sentences_offsets_list)
            ):
                if entity_start_offset >= sent_start_offset and entity_end_offset <= sent_end_offset:
                    sentence = sent
                    found_sentence_num = num
                    rel_start_offset = entity_start_offset - sent_start_offset
                    rel_end_offset = entity_end_offset - sent_start_offset
                    break
            context = ""
            if sentence:
                start_of_sentence = 0
                end_of_sentence = len(sentence)
                if len(sentence) > self.max_text_len:
                    start_of_sentence = max(rel_start_offset - self.max_text_len // 2, 0)
                    end_of_sentence = min(rel_end_offset + self.max_text_len // 2, len(sentence))
                text_before = sentence[start_of_sentence:rel_start_offset]
                text_after = sentence[rel_end_offset:end_of_sentence]
                context = text_before + "[ent]" + text_after
                if self.full_paragraph:
                    cur_sent_len = len(re.findall(self.re_tokenizer, context))
                    first_sentence_num = found_sentence_num
                    last_sentence_num = found_sentence_num
                    context = [context]
                    while True:
                        added = False
                        if last_sentence_num < len(sentences_list) - 1:
                            sentence_tokens = re.findall(self.re_tokenizer, sentences_list[last_sentence_num + 1])
                            last_sentence_len = len(sentence_tokens)
                            if cur_sent_len + last_sentence_len < self.max_paragraph_len:
                                context.append(sentences_list[last_sentence_num + 1])
                                cur_sent_len += last_sentence_len
                                last_sentence_num += 1
                                added = True
                        if first_sentence_num > 0:
                            sentence_tokens = re.findall(self.re_tokenizer, sentences_list[first_sentence_num - 1])
                            first_sentence_len = len(sentence_tokens)
                            if cur_sent_len + first_sentence_len < self.max_paragraph_len:
                                context = [sentences_list[first_sentence_num - 1]] + context
                                cur_sent_len += first_sentence_len
                                first_sentence_num -= 1
                                added = True
                        if not added:
                            break
                    context = " ".join(context)

            log.debug(f"rank, context: {context}")
            contexts.append(context)

        scores_list = self.entity_ranker(contexts, cand_ent_list, cand_ent_descr_list)
        for entity_substr, tag, context, candidate_entities, substr_len, entities_scores, scores in zip(
                entity_substr_list, tags_list, contexts, cand_ent_list, substr_lens, entities_scores_list, scores_list
        ):
            entities_with_scores = []
            max_conn_score = 0
            if scores_dict and scores:
                max_conn_score = max([scores_dict.get(entity, 0) for entity, _ in scores])
            for entity, score in scores:
                substr_score = round(entities_scores.get(entity, (0.0, 0))[0], 2)
                num_rels = entities_scores.get(entity, (0.0, 0))[1]
                if len(context.split()) < 4:
                    score = 0.95
                elif scores_dict and max_conn_score > 0 and scores_dict.get(entity, 0) == max_conn_score:
                    score = 1.0
                    num_rels = 200
                entities_with_scores.append((entity, substr_score, num_rels, score))

            if tag == "t":
                entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[2], x[3]), reverse=True)
            else:
                entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"{entity_substr} --- tag: {tag} --- entities_with_scores: {entities_with_scores}")

            if not entities_with_scores:
                top_entities = []
                top_conf = []
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = []
                top_conf = []
            elif entities_with_scores and (
                    entities_with_scores[0][1] < 0.3
                    or (entities_with_scores[0][3] < 0.13 and entities_with_scores[0][2] < 20)
                    or (entities_with_scores[0][3] < 0.3 and entities_with_scores[0][2] < 4)
                    or entities_with_scores[0][1] < 0.6
            ):
                top_entities = []
                top_conf = []
            else:
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

            high_conf_entities = []
            high_conf_nums = []
            for elem_num, (entity, conf) in enumerate(zip(top_entities, top_conf)):
                if len(conf) == 3 and conf[0] >= 1.0 and conf[1] > 50 and conf[2] > 0.3:
                    new_conf = list(conf)
                    if new_conf[1] > 55:
                        new_conf[2] = 1.0
                    new_conf = tuple(new_conf)
                    high_conf_entities.append((entity,) + new_conf)
                    high_conf_nums.append(elem_num)

            high_conf_entities = sorted(high_conf_entities, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"high_conf_entities: {high_conf_entities}")
            for n, elem_num in enumerate(high_conf_nums):
                if elem_num - n >= 0 and elem_num - n < len(top_entities):
                    del top_entities[elem_num - n]
                    del top_conf[elem_num - n]

            top_entities = [elem[0] for elem in high_conf_entities] + top_entities
            top_conf = [elem[1:] for elem in high_conf_entities] + top_conf
            
            if self.sort_low_conf:
                top_entities, top_conf = self.sort_out_low_conf(entity_substr, top_entities, top_conf)

            if not top_entities:
                entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[2], x[3]), reverse=True)
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

            if self.num_entities_to_return == 1 and top_entities:
                entity_ids_list.append(top_entities[0])
                conf_list.append(top_conf[0])
            elif self.num_entities_to_return == "max":
                if top_conf:
                    max_conf = top_conf[0][0]
                    max_rank_conf = top_conf[0][2]
                    entity_ids, confs = [], []
                    for entity_id, conf in zip(top_entities, top_conf):
                        if (conf[0] >= max_conf * 0.9 and max_rank_conf <= 1.0) \
                                or (max_rank_conf == 1.0 and conf[2] == 1.0):
                            entity_ids.append(entity_id)
                            confs.append(conf)
                    entity_ids_list.append(entity_ids)
                    conf_list.append(confs)
                else:
                    entity_ids_list.append([])
                    conf_list.append([])
            else:
                entity_ids_list.append(top_entities[: self.num_entities_to_return])
                conf_list.append(top_conf[: self.num_entities_to_return])
            log.debug(f"{entity_substr} --- top entities {entity_ids_list[-1]} --- top_conf {conf_list[-1]}")
        return entity_ids_list, conf_list

    def sort_out_low_conf(self, entity_substr, top_entities, top_conf):
        if len(entity_substr.split()) > 1 and top_conf:
            f_top_entities, f_top_conf = [], []
            for top_conf_thres, conf_thres in [(1.0, 0.9), (0.9, 0.8)]:
                if top_conf[0][0] >= top_conf_thres:
                    for ent, conf in zip(top_entities, top_conf):
                        if conf[0] > conf_thres:
                            f_top_entities.append(ent)
                            f_top_conf.append(conf)
            return f_top_entities, f_top_conf
        return top_entities, top_conf

    def rank_by_connections(self, ids_list):
        objects_sets_dict, scores_dict, conn_dict = {}, {}, {}
        for ids in ids_list:
            for entity_id in ids:
                scores_dict[entity_id] = 0
                conn_dict[entity_id] = set()
        for ids in ids_list:
            for entity_id in ids[:self.num_entities_for_conn_ranking]:
                objects = set()
                for prefix in self.prefixes["entity"]:
                    tr, _ = self.kb.search_triples(f"{prefix}/{entity_id}", "", "")
                    for subj, rel, obj in tr:
                        if rel.split("/")[-1] not in {"P31", "P279"}:
                            if any([obj.startswith(pr) for pr in self.prefixes["entity"]]):
                                objects.add(obj.split("/")[-1])
                            if rel.startswith(self.prefixes["rels"]["no_type"]):
                                tr2, _ = self.kb.search_triples(obj, "", "")
                                for _, rel2, obj2 in tr2:
                                    if rel2.startswith(self.prefixes["rels"]["statement"]) \
                                            or rel2.startswith(self.prefixes["rels"]["qualifier"]):
                                        if any([obj2.startswith(pr) for pr in self.prefixes["entity"]]):
                                            objects.add(obj2.split("/")[-1])
                objects_sets_dict[entity_id] = objects
                for obj in objects:
                    if obj not in objects_sets_dict:
                        objects_sets_dict[obj] = set()
                    objects_sets_dict[obj].add(entity_id)

        for i in range(len(ids_list)):
            for j in range(len(ids_list)):
                if i != j:
                    for entity_id1 in ids_list[i][:self.num_entities_for_conn_ranking]:
                        for entity_id2 in ids_list[j][:self.num_entities_for_conn_ranking]:
                            if entity_id1 in objects_sets_dict[entity_id2]:
                                conn_dict[entity_id1].add(entity_id2)
                                conn_dict[entity_id2].add(entity_id1)
        for entity_id in conn_dict:
            scores_dict[entity_id] = len(conn_dict[entity_id])
        return scores_dict
