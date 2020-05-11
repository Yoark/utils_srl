from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance

@Predictor.register('bound_predictor')
class BoundPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        outputs['tokens'] = [str(token) for token in instance.fields['tokens']]
        # i should compute this from class prob
        outputs['predict'] = [label_vocab[i] for i in outputs['logits'].argmax(1)]
        outputs['gold'] = instance.fields['tags'].labels
        # import ipdb; ipdb.set_trace()
        outputs['metadata'] = instance.fields['metadata'].metadata
        return sanitize(outputs)


