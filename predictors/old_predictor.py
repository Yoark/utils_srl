from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance

@Predictor.register('old_predictor')
class OldPredictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        outputs['tokens'] = [str(token) for token in instance.fields['tokens']]
        outputs['predict'] = [label_vocab[i] for i in outputs['logits'].argmax(1)]
        outputs['gold'] = instance.fields['tags'].labels
        outputs['metadata'] = instance.fields['metadata'].metadata
        
        return sanitize(outputs)