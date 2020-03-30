#!/usr/bin/env python
# coding: utf-8
from allennlp.data.dataset_readers import SrlReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from pytorch_pretrained_bert.tokenization import BertTokenizer
from allennlp.data.tokenizers import Token
import logging
from typing import Dict, List, Iterable, Tuple, Any
from allennlp.common.file_utils import cached_path
from allennlp.models import srl_bert
import json
import numpy as np
import torch
from allennlp.data.dataset_readers.semantic_role_labeling import _convert_verb_indices_to_wordpiece_indices
from allennlp.data.dataset_readers.semantic_role_labeling import _convert_tags_to_wordpiece_tags

from ..utils import load_obj_tsv
from typing import Iterable, Iterator, Callable
import logging
import os
import pathlib

import jsonpickle

from allennlp.common import Tqdm, util
from allennlp.common.checks import ConfigurationError
from allennlp.common.registrable import Registrable
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("bound_image_srl")
class BoundSrlReader(SrlReader):

    def __init__(self, token_indexers=None, domain_identifier=None, lazy=False, bert_model_name=None):
        super().__init__(token_indexers=token_indexers, domain_identifier=domain_identifier, lazy=lazy, bert_model_name=bert_model_name)
        # ! here 's test:
        # self.train_file = '/home/zijiao/research/data/Flicker/mscoco_imgfeat/data_created/train/imgs.tsv'
        # self.val_file = '/home/zijiao/research/data/Flicker/mscoco_imgfeat/data_created/val/imgs.tsv'
        # self.train_imgid2img = self.load_feature(self.train_file)
        # self.val_imgid2img = self.load_feature(self.val_file)
        # self.imgid2img = {}
        # self.img_file = ''
        # self.text_file ='' 
        # self.folder_path = '' 

    def load_feature(self, file_path):
        imgid2img = {}
        self.features = load_obj_tsv(file_path)
        for img_datum in self.features:
            imgid2img[img_datum['img_id']] = img_datum
        return imgid2img

    def read(self, file_path: str) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        # self.folder_path = file_path
        img_file = os.path.join(file_path,'imgs.tsv')
        imgid2img = self.load_feature(img_file)

        lazy = getattr(self, 'lazy', None)

        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")

        if self._cache_directory:
            cache_file = self._get_cache_location_for_file_path(file_path)
        else:
            cache_file = None

        if lazy:
            return _LazyInstances(lambda: self._read(file_path, imgid2img),
                                  cache_file,
                                  self.deserialize_instance,
                                  self.serialize_instance)
        else:
            # First we read the instances, either from a cache or from the original file.
            if cache_file and os.path.exists(cache_file):
                instances = self._instances_from_cache_file(cache_file)
            else:
                instances = self._read(file_path)

            # Then some validation.
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))

            # And finally we write to the cache if we need to.
            if cache_file and not os.path.exists(cache_file):
                logger.info(f"Caching instances to {cache_file}")
                self._instances_to_cache_file(cache_file, instances)

            return instances

    def _read(self, file_path, imgid2img):
        #file_path = cached_path(text_path)
        #logger.info(f"Reading SRL instances fro dataset files at:{file_path}")
        # self.folder_path = folder_path
        text_file = os.path.join(file_path, 'srls.json')
        # ! This is commendted out for experiemnt
        # if 'val' in folder_path.split('/'):
        #     self.imgid2img = self.val_imgid2img
        # else:
        #     self.imgid2img = self.train_imgid2img
        # img_file = folder_path + 'imgs.tsv'
        #if img_path:
        # img_features = load_obj_tsv(img_file)
        # for img_datum in img_features:
            # self.imgid2img[img_datum['img_id']] = img_datum

        with open(text_file) as f:
            quples = json.loads(f.read())
            for srl in quples:
                tokens = [Token(t) for t in srl["caption"].split()]
                tags = srl["tag"].split()
                # verb_indicator = [0] * len(tags)
                # verb_indicator["predicate"] = 1
                verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]

                # remove .jpg

                # 
                # import ipdb; ipdb.set_trace()
                img = imgid2img[srl["image"][:-4]]
                yield self.text_to_instance(tokens, verb_indicator, img, tags)



    def text_to_instance(  # type: ignore
        self, tokens: List[Token], verb_label: List[int], img, tags: List[str] = None
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
            #? Maybe other options???
            img_feats = img['features'].copy()
            img_boxes = img['boxes'].copy()
            obj_num = img['num_boxes']
            assert len(img_feats) == len(img_boxes) == obj_num

            # Normalize the boxes to 0 ~ 1
            img_boxes = img_boxes.copy()
            img_boxes[:, (0, 2)] /= img['img_w']
            img_boxes[:, (1, 3)] /= img['img_h']
            np.testing.assert_array_less(img_boxes, 1+1e-5)
            np.testing.assert_array_less(-img_boxes, 0+1e-5)

        # Concat box feats to each object features
        img_concat = np.hstack((img_feats, img_boxes))
        img_field = ArrayField(img_concat)

        fields: Dict[str, Field] = {}
        fields["tokens"] = text_field
        fields["verb_indicator"] = verb_indicator
        fields["img_emb"] = img_field

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

