# Label-Free Handwritten Text Recognition Learner 

It is a toolkit providing 
a set of tools for training a neural net to recognize handwritten words.
No labeled data is needed. The toolkit achieves this in two phases:
- pre-train the model on synthetic handwriting images
- fine-tune the model on a dataset of real unlabeled handwriting images via a self-training technique.

This implementation is based on the following paper: [Self-Training of Handwritten Word Recognition for
Synthetic-to-Real Adaptation](https://arxiv.org/abs/2206.03149).

# Possible use cases
Offline Handwritten Word Recognition converts an image that contains a handwritten word 
into a machine-readable representation (text). So it is Image-to-Text technology.
A pre-trained handwritten word/line recognizer usually gets used as a part of a complete 
pipeline. For example, the full-page recognition system will apply handwritten text 
segmentation to split the page image into lines and individual words. Then, a 
handwritten word recognition neural net will transcribe these tightly cropped word images into text.

# Why use this toolkit
Training a neural net for handwriting recognition requires large amounts of handwriting data. 
Collecting the data is a laborious and time-consuming process. On the other hand, generating 
synthetic handwritten images is cheap due to the abundance of handwritten fonts.

It turns out that training the model on purely synthetic images already gives a decent performance. 
A follow-up synthetic-to-real adaptation (fine-tuning) can significantly improve the accuracy of 
the final system. It can also help the model adapt to the style of a particular writer.

The toolkit enables one to build a handwriting recognizer using only unlabeled data. Moreover, 
pretraining on purely synthetic images requires no data at all.

# What can it do

Concretely, the toolkit implements the following tasks:
- extracting probability distribution from text corpora
- generating synthetic handwritten word images
- training on synthetic images
- fine-tuning on real (handwritten word) images
- evaluating a pretrained model by computing metrics (loss, CER, WER, etc.)
- using the pretrained model to recognize handwritten words
- saving/retrieving model checkpoints 

# What is needed
To make it work, the toolkit expects a developer to provide:
- handwritten fonts
- text corpora
- real unlabeled images of handwritten words
- optionally, real labeled images of handwritten words (see the note)

Note: although transcriptions aren't required, a small subset of transcribed/labeled 
images will help estimate metrics.

# Documentation

You can find a documentation at [here](https://github.com/X-rayLaser/label-free-handwritten-text-recognition-learner/wiki) 

# Support

If you find this repository useful, consider starring it by clicking at the ★ button. It will be much appreciated.
