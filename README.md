[toc]

# Plan

## Objective

- [x] Build multimodel that read in image + caption, outputs: srl labels for captions
- [ ] Investigate the interaction between image and caption (mainly observe the attention learned)
- [ ] Iterate

## Dataloader
- [x] try naive solution currently have
- [x] Solve the memory problem, by make it **lazy**
- [ ] use lxmert solution, that loads data with two class interaction
- [ ] Optimize the image feature reading step./ **half way, buggy**
## Dataset
- [x] Create val, test set for mscoco for current model
- [x] Modify the model before (which only use alignment score as the interaction channel between image and caption) that could take in current data format / **half way**
- [ ] Create Flicker test dataset 
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

### Attention
- [x] naive top down attention /half way
- [x] Top down attention combined with alignment score/ **half way**
- [ ] co-attention
## Results

- 0.885 f1 score for validation for attention model
- 0.884 for bilstm and highway, 0.877 for alignment model. **But this comparision is based on a different data split.** The validation data used now is a lot larger than before
  - Now: 

- [ ] Perform evaluation on Flicker test dataset

  

## Analysis
- [x] Check the model training, especially the decoding part.
- [ ] check the obtained mapping between img feat and text entities/(semantic roles)
- [ ] visualize
- [ ] Ablations : attention + others, compares to others. to investigate how much image helps
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

  

