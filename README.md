# Plan
## Dataloader
- [x] try naive solution currently have
- [x] Solve the memory problem, by make it **lazy**
- [ ] use lxmert solution, that loads data with two class interaction
## Dataset
- [ ] Create val, test set for mscoco
- [ ] Create Flicker dataset
## Attention
- [x] naive top down attention /half way
- [ ] co-attention
## Analysis
- [ ] Check the model training, especially the decoding part.
- [ ] check the obtained mapping between img feat and text entities/(semantic roles)
- [ ] visualize
## Test
- [x] Create toy dataset for debugging
- [x] fix the ipdb bug
- [x] create dubug file
- [x] clean the model and loader modules
NEED TO MAKE LOADING MORE EFFICIENT, currently reread overly
