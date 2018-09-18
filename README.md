Pytorch Implementation of Improving Reinforcement Learning Based Image Captioning with Natural Language Prior

## Requirements
Python 2.7 

PyTorch 0.4 (along with torchvision)

cider package （copy from [Here](https://drive.google.com/open?id=15jqeHYQD0LJjp_e86QvJipUL4_-MHH5p) and dump them to `cider/`)

pycoco package (copy from [Here](https://drive.google.com/open?id=1B71eCxPj8h7cw5SGVyKOLPsjbbM6dFAF) and extract them to `pycoco/`)

You need to download pretrained resnet model for both training and evaluation. The models can be downloaded from [here](https://drive.google.com/open?id=1YD7YjPPoK-WGZhmeTcV8LEp_3hYoBcpq), and should be placed in `data/imagenet_weights`.

## Train your own network on COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](https://drive.google.com/open?id=1ZmAqqknqPVnwmiPS2KF6wQCURVhTuZp2) following Karpathy's split. Copy `dataset_coco.json`,`captions_train.json`,`captions_val.json` and `captions_test.json` in to `data/features`. 

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

### Download COCO dataset and pre-extract the image features 

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.

Then:

```
$ python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```


`prepro_feats.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

### Warm Starm

In order to help CIDEr based REINFORCE algorithm converge more stable and faster, We need to warm start the captioning model and run the script below 

```bash
$ python train_warm.py --caption_model fc 
```
if you want to use Attention, then run 
```bash
$ python train_warm.py --caption_model att 
```
Download our pretrained warm start model from this [link](https://drive.google.com/open?id=1ZmAqqknqPVnwmiPS2KF6wQCURVhTuZp2). And the best CIDEr score in validation set are 90.1 for FC and 94.2 for Attention.

### Train using Self-critical 
```bash
$ python train_sc_cider.py --caption_model att 
```
You will see a large boost of CIDEr score but with lots of bad endings.
![Image text](https://github.com/tgGuo15/PriorImageCaption/blob/master/images/badending.png)



### Train using Ngram constraint

First you should preprocess the dataset and get the ngram data:
```
$ python get_ngram.py
```
and will generate `fourgram.pkl` and `trigram.pkl` in `data/` .

Then
```bash
$ python train_fourgram.py  --caption_model fc 
```
It will take almost 40,000 iterations to converge and the experiment details are written in `experiment.log` in `save_dir` like 
![Image text](https://github.com/tgGuo15/PriorImageCaption/blob/master/images/fourgram_att.png)


### Train using Neural Language model

First you should train a neural language or you can download our pretrained LSTM language model from [link](https://drive.google.com/open?id=1ZmAqqknqPVnwmiPS2KF6wQCURVhTuZp2).
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
It will take almost 36,000 iterations to converge and the experiment details are written in `experiment.log` in `save_dir`.

![Image text](https://github.com/tgGuo15/PriorImageCaption/blob/master/images/rnn_att.png)


### Evaluating `CIDEr`,`METEOR`,`ROUGEL`,`BLEU`score with Bad Ending removal
```bash
$ python Eval_model.py  --caption_model fc --rl_type fourgram
```

### Try another network structure
We also try another neural network structure and get the similar results. Please see the MoreNet.md for more details. 

## Acknowledgements
Thanks the original [self-critical](https://github.com/ruotianluo/self-critical.pytorch) performed by ruotianluo.
