We use the same data preprocess as in README.md and copy all files in `MoreNet/` to replace in the main directory.
### Warm Starm
In order to help CIDEr based REINFORCE algorithm converge more stable and faster, We need to warm start the captioning model and run the script below 

```bash
$ python train_warm.py --caption_model fc 
```
if you want to use Attention, then run 
```bash
$ python train_warm.py --caption_model att 
```
Download our pretrained warm start model from this [link](https://drive.google.com/open?id=1fj_Dgy9Gmxc9t6phzWKaH6DUZZXB-a6T). 

### Train using Self-critical 
```bash
$ python train_sc_cider.py --caption_model att 
```
You will also see a large boost of CIDEr score but with lots of bad endings.

### Train using Ngram constraint
```bash
$ python train_fourgram.py  --caption_model fc 
```

### Train using Neural Language model

First you should train a neural language or you can download our pretrained LSTM language model from [link](https://drive.google.com/open?id=1fj_Dgy9Gmxc9t6phzWKaH6DUZZXB-a6T).
```
$ python train_rnnlm.py
```

Then train RL setting with Neural Language model constraint with the same warm start model.
```bash
$ python train_rnnlm_cider.py  --caption_model fc 
```
or 
```bash
$ python train_rnnlm_cider.py  --caption_model att 
```

