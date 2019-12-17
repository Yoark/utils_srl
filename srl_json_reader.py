#!/usr/bin/env python
# coding: utf-8


# TODO:
# - [x] Change datareader to cornll05 train set formmat
# - [x]  run model for test:
# - [x] change model to accept json train test data
# - [ ] test model
# - [ ] adding image attenter into the model
# - [ ] see performance
# - [ ] make some results:



ontonotes_datapath = '/media/data/srl/srl/conll-formatted-ontonotes-5.0/'




import os
from allennlp.data.dataset_readers import SrlReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.data.tokenizers import Token
import logging
from typing import Dict, List, Iterable, Tuple, Any
from allennlp.common.file_utils import cached_path
from allennlp.models import srl_bert
import json






def _read(file_path):
    file_path = cached_path(file_path)
    #logger.info(f"Reading SRL instances fro dataset files at:{file_path}")
    with open(file_path) as f:
        for line in f.readlines():
            srl = json.loads(line)
            tokens = [Token(t) for t in srl["words"]]
            # to test module, need more work
            if srl['verbs']:
                tags = srl['verbs'][0]['tags']
            else:
                continue
            verb_indictor = [1 if label[-2:] == "-V" else 0 for label in tags]
            yield self.tokens, verb_indictor, tags



@DatasetReader.register("image_srl")
class ImageSrlReader(SrlReader):
    def _read(self, file_path):
        file_path = cached_path(file_path)
        #logger.info(f"Reading SRL instances fro dataset files at:{file_path}")
        with open(file_path) as f:
            for line in f.readlines():
                srl = json.loads(line)
                tokens = [Token(t) for t in srl["words"]]
                if srl['verbs']:
                    tags = srl['verbs'][0]['tags']
                else:
                    continue
                verb_indictor = [1 if label[-2:] == "-V" else 0 for label in tags]
                yield self.text_to_instance(tokens, verb_indictor, tags)






