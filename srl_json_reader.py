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
import numpy as np





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


# TODO add image into reader
@DatasetReader.register("json_srl")
class jsonSrlReader(SrlReader):
    
    def _read(self, file_path):
        file_path = cached_path(file_path)
        #logger.info(f"Reading SRL instances fro dataset files at:{file_path}")
        with open(file_path) as f:
            for line in f.readlines():
                srl = json.loads(line)
                tokens = [Token(t) for t in srl["words"]]
                if not srl['verbs']:
                # Sentence contains no predicates.
                    tags = ["O" for _ in tokens]
                    verb_label = [0 for _ in tokens]
                    yield self.text_to_instance(tokens, verb_label, tags)
                else:   
                    for verb_dict in srl['verbs']:
                        verb_indicator = [1 if label[-2:] == "-V" else 0 for label in verb_dict['tags']]
                        tags = verb_dict['tags']
                        yield self.text_to_instance(tokens, verb_indictor, tags)


@DatasetReader.register("image_srl")
class ImageSrlReader(SrlReader):
    
    def _read(self, text_path, img_path=None):
        file_path = cached_path(text_path)
        #logger.info(f"Reading SRL instances fro dataset files at:{file_path}")
        if img_path:
            img_embs = np.load(img_path)
        with open(file_path) as f:
            for i, line in enumerate(f.readlines()):
                srl = json.loads(line)
                tokens = [Token(t) for t in srl["words"]]
                # ? maybe put it somewhere else
                img_id = i // 5
                img_emb = img_embs[img_id]
                if not srl['verbs']:
                # Sentence contains no predicates.
                    tags = ["O" for _ in tokens]
                    verb_label = [0 for _ in tokens]
                    yield self.text_to_instance(tokens, verb_label, img_emb, tags)
                else:   
                    for verb_dict in srl['verbs']:
                        verb_indicator = [1 if label[-2:] == "-V" else 0 for label in verb_dict['tags']]
                        tags = verb_dict['tags']
                        yield self.text_to_instance(tokens, verb_indicator, img_emb, tags)



    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], image_imb, tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(
                [t.text for t in tokens]
            )
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                token_indexers=self._token_indexers,
            )
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator

        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields["tags"] = SequenceLabelField(new_tags, text_field)
            else:
                fields["tags"] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
