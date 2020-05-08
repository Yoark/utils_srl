[toc]

# Plan



## This Week

- [ ] Check if the image one could help alleviate the neccessity of decoding, concretely, decrease the need of hard constraints, like no two identical proto roles for the same predicate.

- [ ] The evaluation metrics

- [ ] Results of average F1, (small sample, 5, 20 epochs)

  - [ ] ```
    attn :0.7534545065112997 , old : 0.7434289401331587  text: 0.7500537854210754
    ```

- [ ] Check the attention part

- [ ] Paper review writing

- [ ] Before and after decoding comparision,

  - [ ] image may 

- [ ] Argument level confusion matrix

## Past

- [x] Cleaned and Create over 5000 flicker data points ready to use for fine tuning or evaluation

- [x] Train on more epochs (50), achieve 0.7572, 0.7395 for new, old model on flicker evaluation set (3127), also pure text model 0.7525.

  - [x] | attention+bilstm, img bounding boxes+text | alignment+bilstm, img+text | text   |
    | :---------------------------------------: | -------------------------- | ------ |
    |                  0.7572                   | 0.7395                     | 0.7525 |

- [x] Fixed model bug

- [x] Create resnet features for Flicker dataset and converted flicker evaluation data to old model format

- [x] Ask About removing some examples from the Flicker dataset.
- [x] Another loader bug!!!
- [x] ==Data source==
- [x] Danny dag (Oregon State uNiv)
- [x] Fine tuning (on going)
- [ ] Co-attention
- [ ] alignment-attention model
- [x] Check what gained for new model VS old model VS pure text model by inspecting errors
(on going) error rate decrease for ARG-LOC,:
- [ ] Statistical fidility (on going) trained 2 models for each setting, need more
- [ ] Boostrapping sampling
## Objective

- [x] Build multimodel that read in image + caption, outputs: srl labels for captions
- [ ] Investigate the interaction between image and caption (mainly observe the attention learned)
- [ ] Check if the image one could help alleviate the neccessity of decoding, concretely, decrease the need of hard constraints, like no two identical proto roles for the same predicate.
- [ ] Iterate
- [x] Confusion matrix
- [o] Per predicate consistency between performance and frequency
- [ ] exclude unexistable 

## Dataloader
- [x] try naive solution currently have
- [x] Solve the memory problem, by make it **lazy**
- [ ] use lxmert solution, that loads data with two class interaction
- [x] Optimize the image feature reading step.
## Dataset
- [x] Create val, test set for mscoco for current model
- [x] Modify the model before (which only use alignment score as the interaction channel between image and caption) that could take in current data format / **half way**
- [x] Create Flicker test dataset 
- [ ] Validate that the two dataset for two models contains same entities.
## Model:

### Parameters:

``````json
{
  "dataset_reader":{"type":"bound_image_srl",
                    "lazy": true
                   },
  "train_data_path": '/home/zijiao/research/data/Flicker/mscoco_imgfeat/data_created/train/',
  "validation_data_path": '/home/zijiao/research/data/Flicker/mscoco_imgfeat/data_created/val/',
  "model": {
    "type": "box_image_srl",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "/home/zijiao/research/Thesis/data/glove.6B.100d.txt.gz",
            "trainable": true
        }
      }
    },
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 200,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
    "binary_feature_dim": 100
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },

  "trainer": {
    "num_epochs": 10,
    "grad_clipping": 1.0,
    "patience": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 0,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  }
}
``````

### Training
- [x] train every model for 50 epochs.
- [ ] Do meta learning ?? to transfer mscoco to generate on Flikcer??
- [ ] few shot learning?
- [ ] Transferring?
- [ ] Flikcer has at least 2k available data to use. fine tune on that ????
- [ ] 

### Attention

- [x] naive top down attention /half way
- [x] Top down attention combined with alignment score/ **half way**
- [ ] co-attention
- [ ] Combine the **alignment score** and **attention**
## Results

- 0.885 f1 score for validation for attention model
- 0.884 for bilstm and highway, 0.877 for alignment model. **But this comparision is based on a different data split.** The validation data used now is a lot larger than before
  - Now: 
0.823??? bug??? need more epochs?
0.887 for same split of mscoco on old model

$$$ Evaluate 
0.64 on flicker number of sampes can be increase (
0.75 f1 score on 3177 data points.
- [x] Perform evaluation on Flicker test dataset(to make fair comparision between old model and att model, need to generate image feature(whole image) with RCNN ??
- [x] create flicker test dataset for old model?
- [x] Perform evaluation on same splits of mscoco dataset for old model
  

## Analysis
- [x] Check the model training, especially the decoding part.
- [ ] check the obtained mapping between img feat and text entities/(semantic roles)
- [ ] visualize
- [ ] Ablations : attention + others, compares to others. to investigate how much image helps
- [ ] Check with the help of image, how is the error distribution changed.
## Testing
- [x] Create toy dataset for debugging
- [x] fix the ipdb bug
- [x] create dubug file
- [x] clean the model and loader modules
NEED TO MAKE LOADING MORE EFFICIENT, currently reread overly

## Writing

- [ ] **formulate hypos, like could image be help of srl predicting, multimodal information could help build better semantic representation for text**
- [ ] How did they annotate the thing.
  * Include image informaiton helps?
  * or just more acc?
  * integration happen in human mind or in model.
- [ ] Intro
- [ ] RELATED work, 
  * image datasets (image caption like stuff,)
  * srls related work (current srl systems pure text based, object-srl) ==100== Anything srls
  * ==Talk to Jon cai about the block world project.==
- [ ] Model description
- [ ] Data description
- [ ] Experimental setting
- [ ] results
- [ ] Analysis
- [ ] Discussions
  - [ ] Further work suggestion. multiple images from different perps, video. AMR (extra info could be added)
- [ ] Conclusions

## Defense

* Currently plan to be in May, before the end of May (talked to Rajshree)

  * graduating in summer, but won't affect anything as said by some admission officers

## After

- [ ] School selection, about the APs
- [ ] 



