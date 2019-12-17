from typing import Dict, List, TextIO, Optional, Any
from allennlp.nn.util import get_final_encoder_states
import torch
from allennlp.models.model import Model
from allennlp.models import SemanticRoleLabeler
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
import torch.nn as nn
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
import numpy as np
from collections import OrderedDict

import torch.backends.cudnn as cudnn
import torch.nn.init
#import torchvision.models as models
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import functional
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_embeddings, l2norm, cosine_sim, sequence_mask, \
    index_mask, index_one_hot_ellipsis
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import utils
from allennlp.training.metrics.srl_eval_scorer import SrlEvalScorer, DEFAULT_SRL_EVAL_PATH
from allennlp.models.srl_util import (
    convert_bio_tags_to_conll_format,
    write_bio_formatted_tags_to_file,
)
@Model.register("image_srl")
class ImageSRL(SemanticRoleLabeler):

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        binary_feature_dim: int,
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        image_embedding_size: int = 2048,
        lamb: float = 0.2
    ) -> None:
        super().__init__(
            vocab,
            text_field_embedder,
            encoder,
            binary_feature_dim,
            embedding_dropout,
            initializer,
            regularizer,
            label_smoothing,
            ignore_span_metric,
            srl_eval_path,
        )
        
        self.image_embedding_size = image_embedding_size
        self.img_enc = EncoderImagePrecomp(self.image_embedding_size, \
            self.encoder.get_output_dim(),no_imgnorm=False)
        self.vse_loss = ContrastiveLoss(margin=0.2)
        # tune it 
        self.lamb = lamb
        self.lamb = torch.tensor(self.lamb)

    def forward(  # type: ignore
        self,
        tokens: Dict[str, torch.LongTensor],
        verb_indicator: torch.LongTensor,
        img_emb,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        image_embedding: (batch_size, image_embedding_size)
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence and the verb to compute the
            frame for, under 'words' and 'verb' keys, respectively.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat(
            [embedded_text_input, embedded_verb_indicator], -1
        )
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)
        # get final states of shape (batch, embedding_size)
        final_states = get_final_encoder_states(encoded_text, mask)
        # not sure about this
        if torch.cuda.is_available():
            self.img_enc.cuda()
        image_embedding_resized = self.img_enc(img_emb)
        # now compute the alignment loss.

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        if tags is not None:
            seq2seq_loss = sequence_cross_entropy_with_logits(
                logits, tags, mask, label_smoothing=self._label_smoothing
            )
            im_sent_loss = self.vse_loss(final_states, image_embedding_resized)
            
            loss =  seq2seq_loss * (1 - self.lamb) + im_sent_loss.sum() * self.lamb
            #loss = seq2seq_loss
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [
                    example_metadata["verb_index"] for example_metadata in metadata
                ]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get the BIO tags from decode()
                # TODO (nfliu): This is kind of a hack, consider splitting out part
                # of decode() to a separate function.
                batch_bio_predicted_tags = self.decode(output_dict).pop("tags")
                batch_conll_predicted_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                ]
                batch_bio_gold_tags = [
                    example_metadata["gold_tags"] for example_metadata in metadata
                ]
                batch_conll_gold_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                ]
                self.span_metric(
                    batch_verb_indices,
                    batch_sentences,
                    batch_conll_predicted_tags,
                    batch_conll_gold_tags,
                )
            output_dict["loss"] = loss

        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        if metadata is not None:
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
        return output_dict


class EncoderImagePrecomp(nn.Module):
    """ image encoder """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """ Xavier initialization for the fully connected layer """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """ extract image feature vectors """
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images.float())

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)
        
        return features

    def load_state_dict(self, state_dict):
        """ copies parameters, overwritting the default one to
            accept state_dict from Full model """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class ContrastiveLoss(nn.Module):
    """ compute contrastive loss for VSE """
# margin need to be moved to argparse
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        loss_s = (self.margin + scores - d1).clamp(min=0)
        loss_im = (self.margin + scores - d2).clamp(min=0)
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        loss_s = loss_s.masked_fill_(I, 0)
        loss_im = loss_im.masked_fill_(I, 0)

        loss_s = loss_s.mean(1)
        loss_im = loss_im.mean(0)

        return loss_s + loss_im