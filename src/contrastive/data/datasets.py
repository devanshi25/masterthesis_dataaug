import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd

from pathlib import Path
import glob
import gzip
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from sklearn.preprocessing import LabelEncoder
import collections

from pdb import set_trace

def assign_clusterid(identifier, cluster_id_dict, cluster_id_amount):
    try:
        result = cluster_id_dict[identifier]
    except KeyError:
        result = cluster_id_amount
    return result

def serialize_sample_lspc(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split(" ")[:5])}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split(" ")[:50])}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split(" ")[:100])}'.strip()
    string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent"].split(" ")[:200])}'.strip()

    return string

def serialize_sample_abtbuy(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"name"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_amazongoogle(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_walmartamazon(sample):
    string = ''
    string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand"].split())}'.strip()
    string = f'{string} [COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] modelno [VAL] {" ".join(sample[f"modelno"].split())}'.strip()
    string = f'{string} [COL] category [VAL] {" ".join(sample[f"category"].split())}'.strip()
    string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price"]).split())}'.strip()
    string = f'{string} [COL] description [VAL] {" ".join(sample[f"description"].split()[:100])}'.strip()

    return string

def serialize_sample_dblpscholar(sample):
    string = ''
    string = f'{string}[COL] title [VAL] {" ".join(sample[f"title"].split())}'.strip()
    string = f'{string} [COL] authors [VAL] {" ".join(str(sample[f"authors"]).split())}'.strip()
    string = f'{string} [COL] venue [VAL] {" ".join(str(sample[f"venue"]).split())}'.strip()
    string = f'{string} [COL] year [VAL] {" ".join(str(sample[f"year"]).split())}'.strip()

    return string

def serialize_sample_beeradvoratebeer(sample):
    string = ''
    string = f'{string}[COL] Beer_Name [VAL] {" ".join(sample[f"Beer_Name"].split())}'.strip()
    string = f'{string} [COL] Brew_Factory_Name [VAL] {" ".join(sample[f"Brew_Factory_Name"].split())}'.strip()
    string = f'{string} [COL] Style [VAL] {" ".join(sample[f"Style"].split())}'.strip()
    string = f'{string} [COL] ABV [VAL] {" ".join(sample[f"ABV"].split()[:100])}'.strip()

    return string

def serialize_sample_company(sample):
    string = ''
    string = f'{string}[COL] content [VAL] {" ".join(sample[f"content"].split())}'.strip()

    return string

class Augmenter():
    def __init__(self, aug, aug_prob, combo=False):

        self.combo = combo
        stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=aug_prob, aug_word_p=aug_prob)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=aug_prob)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=aug_prob)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=aug_prob)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=aug_prob)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=aug_prob)
        # aug_synonym = naw.SynonymAug(aug_src="wordnet", stopwords=stopwords, aug_p=aug_prob)
        aug_spell = naw.SpellingAug(dict_path='/ceph/dmittal/di-student/spelling_en.txt', stopwords=stopwords, aug_p=aug_prob)
        aug_ocr = nac.OcrAug(stopwords=stopwords, aug_char_p=aug_prob, aug_word_p=aug_prob)
        aug_insert = nac.RandomCharAug(action="insert", stopwords=stopwords, aug_char_p=aug_prob, aug_word_p=aug_prob)
        aug_contextsubstitute = naw.ContextualWordEmbsAug(model_path='distilroberta-base', action="substitute", stopwords=stopwords, aug_p=aug_prob)
        aug_flowswapsubstitute = naf.Sequential([naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=aug_prob), naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=aug_prob)])
        aug_flowseqsubdelete = naf.Sequential([naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=aug_prob), naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=aug_prob)])
        aug_flowsmtcropsubdeleteswap = naf.Sometimes([naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=aug_prob), naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=aug_prob), naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=aug_prob), naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=aug_prob)], aug_p=aug_prob)



        aug = aug.strip('-')

        if aug == 'all':
            self.augs = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]

        if aug == 'rand_2combo':
            augs_list = [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None]
            rand_aug_list = random.sample(augs_list,2)
            self.augs = rand_aug_list
        
        if aug == 'swap_substitute_split_combo':
            self.augs = [aug_swap, aug_sub, aug_split, None]

        if aug == 'delete_swap_substitute_combo':
            self.augs = [aug_del, aug_swap, aug_sub, None]

        if aug == 'swap_delete_combo' :
            self.augs = [aug_swap, aug_del, None]
        
        if aug == 'substitute_delete_combo' :
            self.augs = [aug_sub, aug_del, None]

        if aug == 'crop_delete_combo' :
            self.augs = [aug_crop, aug_del, None]

        if aug == 'typo':
            self.augs = [aug_typo, None]

        if aug == 'swap':
            self.augs = [aug_swap, None]

        if aug == 'delete':
            self.augs = [aug_del, None]

        if aug == 'crop':
            self.augs = [aug_crop, None]

        if aug == 'substitute':
            self.augs = [aug_sub, None]

        if aug == 'split':
            self.augs = [aug_split, None]
        
        if aug == 'none':
            self.augs = [None]
        
        if aug == 'crop_Only':
            self.augs = [aug_crop]

        if aug == 'swap_Only':
            self.augs = [aug_swap]
        
        if aug == 'spell':
            self.augs = [aug_spell, None]

        if aug == 'synonym':
            self.augs = [aug_synonym, None]
        
        if aug == 'ocr':
            self.augs = [aug_ocr, None]

        if aug == 'insert':
            self.augs = [aug_insert, None]

        if aug == 'context_substitute':
            self.augs = [aug_contextsubstitute, None]

        if aug == 'flowswapsubstitute':
            self.augs = [aug_flowswapsubstitute]

        if aug == 'flowsmts':
            self.augs = [aug_flowsmtcropsubdeleteswap]

            
    '''
    def apply_aug(self, string):
        aug = random.choice(self.augs)
        if aug is None:
            return string
        else:
            return aug.augment(string)
    '''
    def apply_aug(self, string):
        if self.combo:
            do_aug = random.randint(0,1)
            if do_aug==1:
                for aug in self.augs:
                    if aug is not None:
                        string = aug.augment(string)
                return string
            else:
                return string
        else:   
            aug = random.choice(self.augs)
            if aug is None:
                return string
            else:
                return aug.augment(string)


class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='lspc', only_interm=False, aug=False, aug_prob=0.1, combo=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug
        self.aug_prob = aug_prob

        if self.aug:
            self.augmenter = Augmenter(self.aug, self.aug_prob, combo)

        data = pd.read_pickle(path)
        
        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description'] = ''
                
        if intermediate_set is not None:
            interm_data = pd.read_pickle(intermediate_set)
            if only_interm:
                data = interm_data
            else:
                data = data.append(interm_data)
        
        data = data.reset_index(drop=True)

        data = data.fillna('')
        data = self._prepare_data(data)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        selection = self.data[self.data['labels'] == example['labels']]
        # if len(selection) > 1:
        #     selection = selection.drop(idx)
        pos = selection.sample(1).iloc[0].copy()

        if self.aug:
            example['features'] = self.augmenter.apply_aug(example['features'])
            pos['features'] = self.augmenter.apply_aug(pos['features'])

        return (example, pos)

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc, axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon, axis=1)

        elif self.dataset == 'dblp-scholar':
            data['features'] = data.apply(serialize_sample_dblpscholar, axis=1)

        elif self.dataset == 'beeradvo-ratebeer':
            data['features'] = data.apply(serialize_sample_beeradvoratebeer, axis=1)

        elif self.dataset == 'company':
            data['features'] = data.apply(serialize_sample_company, axis=1)

        label_enc = LabelEncoder()
        data['labels'] = label_enc.fit_transform(data['cluster_id'])

        self.label_encoder = label_enc

        data = data[['features', 'labels']]

        return data

class ContrastivePretrainDatasetDeepmatcher(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='abt-buy', aug=False, aug_prob=0.1, combo=False, split=True, domain_aug=False, d_aug='delete_common_words', d_aug_prob=0.5):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug
        self.aug_prob = aug_prob
        self.domain_aug = domain_aug
        self.d_aug = d_aug
        self.d_aug_prob = d_aug_prob

        if self.aug:
            self.augmenter = Augmenter(self.aug, self.aug_prob, combo)

        data = pd.read_pickle(path)

        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description'] = ''
        
        if clean:
            train_data = pd.read_json(deduction_set, lines=True)
            
            if dataset == 'abt-buy':
                val = pd.read_csv('../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                val = pd.read_csv('../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                val = pd.read_csv('../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'dblp-scholar':
                val = pd.read_csv('../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                val = pd.read_csv('../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'company':
                val = pd.read_csv('../../data/interim/company/company-valid.csv')


            val_set = train_data[train_data['pair_id'].isin(val['pair_id'])]
            val_set_pos = val_set[val_set['label'] == 1]
            val_set_pos = val_set_pos.sample(frac=0.80)
            val_ids = set()
            val_ids.update(val_set['pair_id'])
            
            train_data = train_data[~train_data['pair_id'].isin(val_ids)]
            train_data = train_data[train_data['label'] == 1]
            train_data = train_data.sample(frac=0.80)

            train_data = train_data.append(val_set_pos)

            bucket_list = []
            for i, row in train_data.iterrows():
                left = f'{row["id_left"]}'
                right = f'{row["id_right"]}'
                found = False
                for bucket in bucket_list:
                    if left in bucket and row['label'] == 1:
                        bucket.add(right)
                        found = True
                        break
                    elif right in bucket and row['label'] == 1:
                        bucket.add(left)
                        found = True
                        break
                if not found:
                    bucket_list.append(set([left, right]))
            
            cluster_id_amount = len(bucket_list)
            
            cluster_id_dict = {}
            for i, id_set in enumerate(bucket_list):
                for v in id_set:
                    cluster_id_dict[v] = i
            data = data.set_index('id', drop=False)
            data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
            #data = data[data['cluster_id'] != cluster_id_amount]

            single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

            index = single_entities.index

            if dataset == 'abt-buy':
                left_index = [x for x in index if 'abt' in x]
                right_index = [x for x in index if 'buy' in x]
            elif dataset == 'amazon-google':
                left_index = [x for x in index if 'amazon' in x]
                right_index = [x for x in index if 'google' in x]
            elif dataset == 'walmart-amazon':
                left_index = [x for x in index if 'walmart' in x]
                right_index = [x for x in index if 'amazon' in x]
            elif dataset == 'dblp-scholar':
                left_index = [x for x in index if 'dblp' in x]
                right_index = [x for x in index if 'scholar' in x]
            elif dataset == 'beeradvo-ratebeer':
                left_index = [x for x in index if 'beeradvo' in x]
                right_index = [x for x in index if 'ratebeer' in x]
            elif dataset == 'company':
                left_index = [x for x in index if 'companyA' in x]
                right_index = [x for x in index if 'companyB' in x]
            
            single_entities = single_entities.reset_index(drop=True)
            single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
            single_entities = single_entities.set_index('id', drop=False)
            single_entities_left = single_entities.loc[left_index]
            single_entities_right = single_entities.loc[right_index]
            
            if split:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_right)

            else:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)
                data1 = data1.append(single_entities_right)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_left)
                data2 = data2.append(single_entities_right)

            if intermediate_set is not None:
                interm_data = pd.read_pickle(intermediate_set)
                if dataset != 'lspc':
                    cols = data.columns
                    if 'name' in cols:
                        interm_data = interm_data.rename(columns={'title':'name'})
                    if 'manufacturer' in cols:
                        interm_data = interm_data.rename(columns={'brand':'manufacturer'})
                    interm_data['cluster_id'] = interm_data['cluster_id']+10000

                data1 = data1.append(interm_data)
                data2 = data2.append(interm_data)

            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)

            label_enc = LabelEncoder()
            cluster_id_set = set()
            cluster_id_set.update(data1['cluster_id'])
            cluster_id_set.update(data2['cluster_id'])
            label_enc.fit(list(cluster_id_set))
            data1['labels'] = label_enc.transform(data1['cluster_id'])
            data2['labels'] = label_enc.transform(data2['cluster_id'])

            self.label_encoder = label_enc
                
        data1 = data1.reset_index(drop=True)

        data1 = data1.fillna('')
        data1 = self._prepare_data(data1)

        data2 = data2.reset_index(drop=True)

        data2 = data2.fillna('')
        data2 = self._prepare_data(data2)

        diff = abs(len(data1)-len(data2))

        if len(data1) > len(data2):
            if len(data2) < diff:
                sample = data2.sample(diff, replace=True)
            else:
                sample = data2.sample(diff)
            data2 = data2.append(sample)
            data2 = data2.reset_index(drop=True)

        elif len(data2) > len(data1):
            if len(data1) < diff:
                sample = data1.sample(diff, replace=True)
            else:
                sample = data1.sample(diff)
            data1 = data1.append(sample)
            data1 = data1.reset_index(drop=True)

        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, idx):
        example1 = self.data1.loc[idx].copy()
        selection1 = self.data1[self.data1['labels'] == example1['labels']]
        # if len(selection1) > 1:
        #     selection1 = selection1.drop(idx)
        

        example2 = self.data2.loc[idx].copy()
        selection2 = self.data2[self.data2['labels'] == example2['labels']]
        # if len(selection2) > 1:
        #     selection2 = selection2.drop(idx)
                     
        if self.domain_aug:
            if len(selection1)>1:
                example1['features'], selection1['features'] = domain_augmenter(example1['features'], selection1['features'], d_aug=self.d_aug, d_aug_prob=self.d_aug_prob)
            if len(selection2)>1:
                example2['features'], selection2['features'] = domain_augmenter(example2['features'], selection2['features'], d_aug=self.d_aug, d_aug_prob=self.d_aug_prob)
            if self.d_aug == "delete_common_words":
                selection1['features'], selection2['features'] = delete_common_words(selection1['features'], selection2['features'], d_aug_prob=self.d_aug_prob)
        
        pos1 = selection1.sample(1).iloc[0].copy()
        pos2 = selection2.sample(1).iloc[0].copy()
        #pos1 = random.choice(selection1)
        #pos2 = random.choice(selection2)

        if self.aug:
            example1['features'] = self.augmenter.apply_aug(example1['features'])
            pos1['features'] = self.augmenter.apply_aug(pos1['features'])
            example2['features'] = self.augmenter.apply_aug(example2['features'])
            pos2['features'] = self.augmenter.apply_aug(pos2['features'])

        return ((example1, pos1), (example2, pos2))

    def __len__(self):
        return len(self.data1)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc, axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon, axis=1)

        elif self.dataset == 'dblp-scholar':
            data['features'] = data.apply(serialize_sample_dblpscholar, axis=1)

        elif self.dataset == 'beeradvo-ratebeer':
            data['features'] = data.apply(serialize_sample_beeradvoratebeer, axis=1)

        elif self.dataset == 'company':
            data['features'] = data.apply(serialize_sample_company, axis=1)

        data = data[['features', 'labels']]

        return data

def delete_common_words(sel1, sel2, d_aug_prob=0.5):
    stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']

    list_sel1 = []
    list_sel2 = []

    for s in sel1:
        s1_list = s.split(' ')
        list_sel1 = list_sel1 + s1_list

    for s in sel2:
        s2_list = s.split(' ')
        list_sel2 = list_sel2 + s2_list

    list_sel1 = [item for item in list_sel1 if item not in stopwords]
    list_sel2 = [item for item in list_sel2 if item not in stopwords]
    #print("sel1",list_sel1) 
    #print("sel2",list_sel2)

    common_words = list(set(list_sel1).intersection(list_sel2))
    num_delete_words = int(d_aug_prob*(len(common_words)))
    delete_words = random.sample(common_words, num_delete_words)
    #print("delete_words", delete_words)

    out_sel1 = []
    for s in sel1:
        list_s = s.split(' ')
        for i in range(num_delete_words):
            len_s = len(list_s)
            if len_s>0:
                del_idx = random.randrange(0, len_s)
                list_s.pop(del_idx)

        s_out = ' '.join(list_s)
        out_sel1.append(s_out)

    return out_sel1, sel2

def domain_augmenter(string1, matching_strings, d_aug='delete_common_words', d_aug_prob=0.5):
        
    stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']
   
    ms_word_list = []
    for s in matching_strings:
        s_list = s.split(' ')
        ms_word_list = ms_word_list + s_list
    
    if d_aug == 'copy_random_words':
        string1_list = string1.split(' ')
        ms_word_list_wo_string1 = [item for item in ms_word_list if item not in string1_list and item not in stopwords]
        num_copy_words = int(d_aug_prob*(len(ms_word_list_wo_string1)))
        copy_words = random.sample(ms_word_list_wo_string1, num_copy_words)
       
        for i in range(num_copy_words):
            len_s1 = len(string1_list)
            cp_idx = random.randrange(0, len_s1)
            string1_list.insert(cp_idx, copy_words[i])
        string1 = ' '.join(string1_list)     
    
    if d_aug == 'delete_rand_words':
        string1_list = string1.split(' ')
        ms_word_list_wo_string1 = [item for item in ms_word_list if item not in string1_list and item not in stopwords]
        num_delete_words = int(d_aug_prob*(len(ms_word_list_wo_string1)))
        
        for i in range(num_delete_words):
            len_s1 = len(string1_list)
            if len_s1>0:
                del_idx = random.randrange(0, len_s1)
                string1_list.pop(del_idx)
        string1 = ' '.join(string1_list)     

    
    return string1, matching_strings


class ContrastivePretrainDatasetDeepmatcherDedupSource(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='abt-buy', aug=False, aug_prob=0.1, combo=False, split=True):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset = dataset
        self.aug = aug
        self.aug_prob = aug_prob

        if self.aug:
            self.augmenter = Augmenter(self.aug, self.aug_prob, combo)

        data = pd.read_pickle(path)

        if dataset == 'abt-buy':
            data['brand'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description'] = ''
        
        if clean:
            train_data = pd.read_json(deduction_set, lines=True)
            
            if dataset == 'abt-buy':
                val = pd.read_csv('../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                val = pd.read_csv('../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                val = pd.read_csv('../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'dblp-scholar':
                val = pd.read_csv('../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                val = pd.read_csv('../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'company':
                val = pd.read_csv('../../data/interim/company/company-valid.csv')


            val_set = train_data[train_data['pair_id'].isin(val['pair_id'])]
            val_set_pos = val_set[val_set['label'] == 1]
            val_set_pos = val_set_pos.sample(frac=0.80)
            val_ids = set()
            val_ids.update(val_set['pair_id'])
            
            train_data = train_data[~train_data['pair_id'].isin(val_ids)]
            train_data = train_data[train_data['label'] == 1]
            train_data = train_data.sample(frac=0.80)

            train_data = train_data.append(val_set_pos)

            bucket_list = []
            for i, row in train_data.iterrows():
                left = f'{row["id_left"]}'
                right = f'{row["id_right"]}'
                found = False
                for bucket in bucket_list:
                    if left in bucket and row['label'] == 1:
                        bucket.add(right)
                        found = True
                        break
                    elif right in bucket and row['label'] == 1:
                        bucket.add(left)
                        found = True
                        break
                if not found:
                    bucket_list.append(set([left, right]))
            
            cluster_id_amount = len(bucket_list)
            
            cluster_id_dict = {}
            for i, id_set in enumerate(bucket_list):
                for v in id_set:
                    cluster_id_dict[v] = i
            data = data.set_index('id', drop=False)

            original_cids = data['cluster_id'].copy()

            data['cluster_id'] = data['id'].apply(assign_clusterid, args=(cluster_id_dict, cluster_id_amount))
            #data = data[data['cluster_id'] != cluster_id_amount]

            single_entities = data[data['cluster_id'] == cluster_id_amount].copy()

            index = single_entities.index

            if dataset == 'abt-buy':
                left_index = [x for x in index if 'abt' in x]
                right_index = [x for x in index if 'buy' in x]
            elif dataset == 'amazon-google':
                left_index = [x for x in index if 'amazon' in x]
                right_index = [x for x in index if 'google' in x]
            elif dataset == 'walmart-amazon':
                left_index = [x for x in index if 'walmart' in x]
                right_index = [x for x in index if 'amazon' in x]
            elif dataset == 'dblp-scholar':
                left_index = [x for x in index if 'dblp' in x]
                right_index = [x for x in index if 'scholar' in x]
            elif dataset == 'beeradvo-ratebeer':
                left_index = [x for x in index if 'beeradvo' in x]
                right_index = [x for x in index if 'ratebeer' in x]
            elif dataset == 'company':
                left_index = [x for x in index if 'companyA' in x]
                right_index = [x for x in index if 'companyB' in x]
            
            single_entities = single_entities.reset_index(drop=True)
            single_entities['cluster_id'] = single_entities['cluster_id'] + single_entities.index
            single_entities = single_entities.set_index('id', drop=False)

            single_entities['original_cid'] = original_cids.loc[single_entities.index]

            single_entities_left = single_entities.loc[left_index]

            single_entities_right = single_entities.loc[right_index]
            
            if split:
                data['original_cid'] = original_cids.loc[data.index]

                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)
                for cid in set(data1['original_cid']):
                    entities = data1[data1['original_cid'] == cid].copy()
                    if len(entities) > 1:
                        data1.loc[entities.index, 'cluster_id'] = entities.iloc[0]['cluster_id']

                
                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_right)
                for cid in set(data2['original_cid']):
                    entities = data2[data2['original_cid'] == cid].copy()
                    if len(entities) > 1:
                        data2.loc[entities.index, 'cluster_id'] = entities.iloc[0]['cluster_id']

            else:
                data1 = data.copy().drop(single_entities['id'])
                data1 = data1.append(single_entities_left)
                data1 = data1.append(single_entities_right)

                data2 = data.copy().drop(single_entities['id'])
                data2 = data2.append(single_entities_left)
                data2 = data2.append(single_entities_right)

            if intermediate_set is not None:
                interm_data = pd.read_pickle(intermediate_set)
                if dataset != 'lspc':
                    cols = data.columns
                    if 'name' in cols:
                        interm_data = interm_data.rename(columns={'title':'name'})
                    if 'manufacturer' in cols:
                        interm_data = interm_data.rename(columns={'brand':'manufacturer'})
                    interm_data['cluster_id'] = interm_data['cluster_id']+10000

                data1 = data1.append(interm_data)
                data2 = data2.append(interm_data)

            data1 = data1.reset_index(drop=True)
            data2 = data2.reset_index(drop=True)

            label_enc = LabelEncoder()
            cluster_id_set = set()
            cluster_id_set.update(data1['cluster_id'])
            cluster_id_set.update(data2['cluster_id'])
            label_enc.fit(list(cluster_id_set))
            data1['labels'] = label_enc.transform(data1['cluster_id'])
            data2['labels'] = label_enc.transform(data2['cluster_id'])

            self.label_encoder = label_enc
                
        data1 = data1.reset_index(drop=True)

        data1 = data1.fillna('')
        data1 = self._prepare_data(data1)

        data2 = data2.reset_index(drop=True)

        data2 = data2.fillna('')
        data2 = self._prepare_data(data2)

        diff = abs(len(data1)-len(data2))

        if len(data1) > len(data2):
            if len(data2) < diff:
                sample = data2.sample(diff, replace=True)
            else:
                sample = data2.sample(diff)
            data2 = data2.append(sample)
            data2 = data2.reset_index(drop=True)

        elif len(data2) > len(data1):
            if len(data1) < diff:
                sample = data1.sample(diff, replace=True)
            else:
                sample = data1.sample(diff)
            data1 = data1.append(sample)
            data1 = data1.reset_index(drop=True)

        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, idx):
        example1 = self.data1.loc[idx].copy()
        selection1 = self.data1[self.data1['labels'] == example1['labels']]
        # if len(selection1) > 1:
        #     selection1 = selection1.drop(idx)
        pos1 = selection1.sample(1).iloc[0].copy()

        example2 = self.data2.loc[idx].copy()
        selection2 = self.data2[self.data2['labels'] == example2['labels']]
        # if len(selection2) > 1:
        #     selection2 = selection2.drop(idx)
        pos2 = selection2.sample(1).iloc[0].copy()

        if self.aug:
            example1['features'] = self.augmenter.apply_aug(example1['features'])
            pos1['features'] = self.augmenter.apply_aug(pos1['features'])
            example2['features'] = self.augmenter.apply_aug(example2['features'])
            pos2['features'] = self.augmenter.apply_aug(pos2['features'])

        return ((example1, pos1), (example2, pos2))

    def __len__(self):
        return len(self.data1)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features'] = data.apply(serialize_sample_lspc, axis=1)

        elif self.dataset == 'abt-buy':
            data['features'] = data.apply(serialize_sample_abtbuy, axis=1)

        elif self.dataset == 'amazon-google':
            data['features'] = data.apply(serialize_sample_amazongoogle, axis=1)

        elif self.dataset == 'walmart-amazon':
            data['features'] = data.apply(serialize_sample_walmartamazon, axis=1)

        elif self.dataset == 'dblp-scholar':
            data['features'] = data.apply(serialize_sample_dblpscholar, axis=1)

        elif self.dataset == 'beeradvo-ratebeer':
            data['features'] = data.apply(serialize_sample_beeradvoratebeer, axis=1)

        elif self.dataset == 'company':
            data['features'] = data.apply(serialize_sample_company, axis=1)

        data = data[['features', 'labels']]

        return data

class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, dataset='lspc', aug=False, aug_prob=0.1, combo=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.aug_prob = aug_prob

        if self.aug:
            self.augmenter = Augmenter(self.aug, self.aug_prob, combo)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
        else:
            data = pd.read_json(path, lines=True)
        
        if dataset == 'abt-buy':
            data['brand_left'] = ''
            data['brand_right'] = ''

        if dataset == 'amazon-google' or dataset == 'walmart-amazon':
            data['description_left'] = ''
            data['description_right'] = ''

        data = data.fillna('')

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/computers_valid_{size}.csv')
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                validation_ids = pd.read_csv(f'../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'dblp-scholar':
                validation_ids = pd.read_csv(f'../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                validation_ids = pd.read_csv(f'../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'company':
                validation_ids = pd.read_csv(f'../../data/interim/company/company-valid.csv')
            if self.dataset_type == 'train':
                data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
            else:
                data = data[data['pair_id'].isin(validation_ids['pair_id'])]

        data = data.reset_index(drop=True)

        data = self._prepare_data(data)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()

        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])

        return example

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(self.serialize_sample_lspc, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_lspc, args=('right',), axis=1)
        elif self.dataset == 'abt-buy':
            data['features_left'] = data.apply(self.serialize_sample_abtbuy, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_abtbuy, args=('right',), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(self.serialize_sample_amazongoogle, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_amazongoogle, args=('right',), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(self.serialize_sample_walmartamazon, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_walmartamazon, args=('right',), axis=1)
        elif self.dataset == 'dblp-scholar':
            data['features_left'] = data.apply(self.serialize_sample_dblpscholar, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_dblpscholar, args=('right',), axis=1)
        elif self.dataset == 'beeradvo-ratebeer':
            data['features_left'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('right',), axis=1)
        elif self.dataset == 'company':
            data['features_left'] = data.apply(self.serialize_sample_company, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_company, args=('right',), axis=1)

        data = data[['features_left', 'features_right', 'label']]
        data = data.rename(columns={'label': 'labels'})

        return data

    def serialize_sample_lspc(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split(" ")[:5])}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split(" ")[:100])}'.strip()
        string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent_{side}"].split(" ")[:200])}'.strip()

        return string

    def serialize_sample_abtbuy(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()
        

        return string

    def serialize_sample_amazongoogle(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_walmartamazon(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] modelno [VAL] {" ".join(sample[f"modelno_{side}"].split())}'.strip()
        string = f'{string} [COL] category [VAL] {" ".join(sample[f"category_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_dblpscholar(self, sample, side):
        
        string = ''
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] authors [VAL] {" ".join(sample[f"authors_{side}"].split())}'.strip()
        string = f'{string} [COL] venue [VAL] {" ".join(sample[f"venue_{side}"].split())}'.strip()
        string = f'{string} [COL] year [VAL] {" ".join(str(sample[f"year_{side}"]).split())}'.strip()

        return string

    def serialize_sample_beeradvoratebeer(self, sample, side):
        
        string = ''
        string = f'{string}[COL] Beer_Name [VAL] {" ".join(sample[f"Beer_Name_{side}"].split())}'.strip()
        string = f'{string} [COL] Brew_Factory_Name [VAL] {" ".join(sample[f"Brew_Factory_Name_{side}"].split())}'.strip()
        string = f'{string} [COL] Style [VAL] {" ".join(sample[f"Style_{side}"].split())}'.strip()
        string = f'{string} [COL] ABV [VAL] {" ".join(sample[f"ABV_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_company(self, sample, side):
        
        string = ''
        string = f'{string}[COL] content [VAL] {" ".join(sample[f"content_{side}"].split())}'.strip()

        return string

class BaselineClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=256, dataset='lspc', aug=False, aug_prob=0.1, combo=False, remove_matching_words=False, delete_match_words=False, delete_non_match_words=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('[COL]', '[VAL]'))
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.aug_prob = aug_prob
        self.remove_matching_words = remove_matching_words
        self.delete_match_words = delete_match_words
        self.delete_non_match_words = delete_non_match_words
        
        if self.aug:
            self.augmenter = Augmenter(self.aug, self.aug_prob, combo)

        if dataset == 'lspc':
            data = pd.read_pickle(path)
        else:
            data = pd.read_json(path, lines=True)

        data = data.fillna('')

        if self.dataset_type != 'test':
            if dataset == 'lspc':
                validation_ids = pd.read_csv(f'../../data/raw/wdc-lspc/validation-sets/computers_valid_{size}.csv')
            elif dataset == 'abt-buy':
                validation_ids = pd.read_csv(f'../../data/interim/abt-buy/abt-buy-valid.csv')
            elif dataset == 'amazon-google':
                validation_ids = pd.read_csv(f'../../data/interim/amazon-google/amazon-google-valid.csv')
            elif dataset == 'walmart-amazon':
                validation_ids = pd.read_csv(f'../../data/interim/walmart-amazon/walmart-amazon-valid.csv')
            elif dataset == 'beeradvo-ratebeer':
                validation_ids = pd.read_csv(f'../../data/interim/beeradvo-ratebeer/beeradvo-ratebeer-valid.csv')
            elif dataset == 'dblp-scholar':
                validation_ids = pd.read_csv(f'../../data/interim/dblp-scholar/dblp-scholar-valid.csv')
            elif dataset == 'company':
                validation_ids = pd.read_csv(f'../../data/interim/company/company-valid.csv')
            if self.dataset_type == 'train':
                data = data[~data['pair_id'].isin(validation_ids['pair_id'])]
            else:
                data = data[data['pair_id'].isin(validation_ids['pair_id'])]

        data = data.reset_index(drop=True)

        data = self._prepare_data(data)
        if self.remove_matching_words:
            data = self._remove_matching_words(data)
        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        
        if self.aug:
            example['features_left'] = self.augmenter.apply_aug(example['features_left'])
            example['features_right'] = self.augmenter.apply_aug(example['features_right'])
        if example['label'] == 1:
            if self.delete_match_words:
                example = self._delete_matching_words(example, self.aug_prob)    
            
            if self.delete_non_match_words:
                example = self._delete_non_matching_words(example, self.aug_prob)

        example_tokenized = self.tokenizer(example['features_left'], example['features_right'], padding=False, truncation='longest_first', max_length=self.max_length)
        example_tokenized['label'] = example['label']

        return example_tokenized

    def __len__(self):
        return len(self.data)
    
    def _delete_non_matching_words(self, example, aug_prob=0.2):
        A = example['features_left']
        B = example['features_right']
        A_list = A.split(' ')
        B_list = B.split(' ')
        matched_words = [m for m in B_list if m in A_list]
        matched_words = list(set(matched_words))
       
        aug_del = naw.RandomWordAug(action="delete", stopwords=matched_words, aug_p=aug_prob)
        example['feature_left'] = aug_del.augment(A)
        example['feature_right'] = aug_del.augment(B)

        return example

    def _delete_matching_words(self, example, aug_prob=0.2):
        A = example['features_left']
        B = example['features_right']
        A_list = A.split(' ')
        B_list = B.split(' ')
        matched_words = [m for m in B_list if m in A_list]
        matched_words = list(set(matched_words))

        non_matched_words_A = [item for item in A_list if item not in matched_words]
        non_matched_words_B = [item for item in B_list if item not in matched_words]

        aug_del_A = naw.RandomWordAug(action="delete", stopwords=non_matched_words_A, aug_p=aug_prob)
        aug_del_B = naw.RandomWordAug(action="delete", stopwords=non_matched_words_B, aug_p=aug_prob)

        example['feature_left'] = aug_del_A.augment(A)
        example['feature_right'] = aug_del_B.augment(B)

        return example

    def _remove_matching_words(self, data):
        num_samples = len(data)
        for item in range(1,num_samples):
            #print ('PRE: ',  data['features_left'].iloc[item],  data['features_right'].iloc[item])
            #if data['label'].iloc[item] == 1:
            descL_list = data['features_left'].iloc[item].split(' ')
            descR_list = data['features_right'].iloc[item].split(' ')
            matched_words = [m for m in descR_list if m in descL_list]
            matched_words = list(set(matched_words))

            remove_words = ['[COL]', '[VAL]', 'description', 'title', 'price']
            matched_words = [word for word in matched_words if word not in remove_words]

            # print ('matched_words: ', matched_words)
            descL_list = [word for word in descL_list if word not in matched_words and len(word)>1]
            descR_list = [word for word in descR_list if word not in matched_words and len(word)>1]

            data['features_left'].iloc[item] = ' '.join(descL_list)
            data['features_right'].iloc[item] = ' '.join(descR_list)
            #print ('POST: ', data['features_left'].iloc[item],  data['features_right'].iloc[item])
        return data


    def _prepare_data(self, data):

        if self.dataset == 'lspc':
            data['features_left'] = data.apply(self.serialize_sample_lspc, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_lspc, args=('right',), axis=1)
        elif self.dataset == 'abt-buy':
            data['features_left'] = data.apply(self.serialize_sample_abtbuy, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_abtbuy, args=('right',), axis=1)
        elif self.dataset == 'amazon-google':
            data['features_left'] = data.apply(self.serialize_sample_amazongoogle, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_amazongoogle, args=('right',), axis=1)
        elif self.dataset == 'walmart-amazon':
            data['features_left'] = data.apply(self.serialize_sample_walmartamazon, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_walmartamazon, args=('right',), axis=1)
        elif self.dataset == 'beeradvo-ratebeer':
            data['features_left'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_beeradvoratebeer, args=('right',), axis=1)
        elif self.dataset == 'sblp-scholar':
            data['features_left'] = data.apply(self.serialize_sample_dblpscholar, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_dblpscholar, args=('right',), axis=1)
        elif self.dataset == 'company':
            data['features_left'] = data.apply(self.serialize_sample_company, args=('left',), axis=1)
            data['features_right'] = data.apply(self.serialize_sample_company, args=('right',), axis=1)

        data = data[['features_left', 'features_right', 'label']]

        return data

    def serialize_sample_lspc(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split(" ")[:5])}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split(" ")[:50])}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split(" ")[:100])}'.strip()
        string = f'{string} [COL] specTableContent [VAL] {" ".join(sample[f"specTableContent_{side}"].split(" ")[:200])}'.strip()

        return string

    def serialize_sample_abtbuy(self, sample, side):
        
        string = ''
        string = f'{string}[COL] title [VAL] {" ".join(sample[f"name_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()
        string = f'{string} [COL] description [VAL] {" ".join(sample[f"description_{side}"].split()[:100])}'.strip()
        

        return string

    def serialize_sample_amazongoogle(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"manufacturer_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()

        return string

    def serialize_sample_walmartamazon(self, sample, side):
        
        string = ''
        string = f'{string}[COL] brand [VAL] {" ".join(sample[f"brand_{side}"].split())}'.strip()
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] modelno [VAL] {" ".join(sample[f"modelno_{side}"].split())}'.strip()
        string = f'{string} [COL] category [VAL] {" ".join(sample[f"category_{side}"].split())}'.strip()
        string = f'{string} [COL] price [VAL] {" ".join(str(sample[f"price_{side}"]).split())}'.strip()

        return string

    def serialize_sample_dblpscholar(self, sample, side):
        
        string = ''
        string = f'{string} [COL] title [VAL] {" ".join(sample[f"title_{side}"].split())}'.strip()
        string = f'{string} [COL] authors [VAL] {" ".join(sample[f"authors_{side}"].split())}'.strip()
        string = f'{string} [COL] venue [VAL] {" ".join(sample[f"venue_{side}"].split())}'.strip()
        string = f'{string} [COL] year [VAL] {" ".join(str(sample[f"year_{side}"]).split())}'.strip()

        return string

    def serialize_sample_beeradvoratebeer(self, sample, side):
        
        string = ''
        string = f'{string}[COL] Beer_Name [VAL] {" ".join(sample[f"Beer_Name_{side}"].split())}'.strip()
        string = f'{string} [COL] Brew_Factory_Name [VAL] {" ".join(sample[f"Brew_Factory_Name_{side}"].split())}'.strip()
        string = f'{string} [COL] Style [VAL] {" ".join(sample[f"Style_{side}"].split())}'.strip()
        string = f'{string} [COL] ABV [VAL] {" ".join(sample[f"ABV_{side}"].split()[:100])}'.strip()

        return string

    def serialize_sample_company(self, sample, side):
        
        string = ''
        string = f'{string}[COL] content [VAL] {" ".join(sample[f"content_{side}"].split())}'.strip()

        return string