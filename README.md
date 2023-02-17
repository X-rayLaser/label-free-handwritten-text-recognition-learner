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
To make it work, the toolkit needs:
- handwritten fonts
- text corpora
- real unlabeled images of handwritten words
- optionally, real labeled images of handwritten words (see the note)

Note: although transcriptions aren't required, a small subset of transcribed/labeled 
images will help estimate metrics.

# Prerequisites

Python version 3.8.9 or higher.

# Installation

It is recommended to install the package using a virtual environment, rather than installing it globally.
```
pip install lafhterlearn
```

Alternatively, you can clone the repository, then from within the repository run this command:
```
pip install -r requirements.txt
```

# Quick start

- Under the repository folder create a directory called "fonts" and fill it with TrueType fonts with extensions
otf or ttf
- Similarly, create a directory called "corpora"
- find text corpora and store them in the "corpora" directory as plain text files 
(you can find a few books to download [here](https://www.gutenberg.org/browse/scores/top))
- Build word distribution and save it in the "word_distribution.csv" file with this command:
```
lafhterlearn word_distr word_distribution.csv --corpora-dir corpora --with-freq
```
- Create a directory named "tuning_data"
Within it, create a subdirectory named "unlabeled" and put handwriting image
files (jpg or png) there. Optionally, you can add another subdirectory called "labeled".
It should contain image files whose names (excluding the file extension part) match their transcriptions. For example,
an image could have the name "apple.jpg" (it will be assumed to have the label "apple").
- create a configuration file called my_config.yml and define some settings (these will override defaults),
for instance:
```
image_height: 64
hidden_size: 128
batch_size: 32
```
- create a training session:
```
lafhterlearn make_session --config_file my_config.yml
```
- Start/resume pretraining on synthetic handwriting images:
```
lafhterlearn pretrain session
```
- Fine-tune a model on real handwriting images (possibly unlabeled)
```
lafhterlearn finetune session
```
- Transcribe the image with the real handwritten word:
```
lafhterlearn transcribe session path/to/some/image.png
```
- Evaluate the model and compute metrics:
```
lafhterlearn evaluate session
```

For more details about each step, see the sections below.

# Training scheme briefly

At the core of the toolkit lies a deep neural net. Its architecture is encoder-decoder with attention.
Training consists of 2 separate stages or phases.

During the first stage, the net is pre-trained on synthetic images
of different words rendered using random pseudo-handwritten fonts.

During the second stage, 
the network is fine-tuned on an (unlabeled) dataset of real handwriting
images through a procedure of self-training.
This procedure consists of repeating the following sequence of steps:
- recognize unlabeled images
- keep only confidently recognized images
- create a dataset from these recognized images
- train the neural net on that dataset

# Preparations

Before proceeding, we need to perform a few steps.

## 1. Getting fonts

To pre-train the net on synthetic images of handwritten words, we need to render them using
a large variety of handwritten fonts. The repository ships with a built-in image generator that generates
(image, transcript) pairs. All that is left is to fetch some fonts to use it.

Google fonts is a large repository containing a lot of different fonts, including handwritten ones.
You can get all of them by cloning their repository from Github:
```
git clone https://github.com/google/fonts.git
```

Now, execute the following command to extract only handwritten fonts.
In this case, it will extract up to 250 handwritten fonts with Latin charset and save them to the "fonts" directory:
```
lafhterlearn extract_fonts path/to/google_fonts/repository fonts --num-fonts 250 --lang latin
```

Alternatively, you can manually create a "fonts" directory and fill it with custom fonts.

## 2. Dictionary and word distribution

Another thing that we need for the image generator is either a dictionary of words or a word distribution table.
A dictionary should be a plain text file consisting of multiple lines, one word per line.
Word distribution should be a file in CSV format in which each line consists of a word and its
probability. The next optional section gives a few more details.

If you have a directory containing text corpora as plain text files,
you can build word distribution with this command:
```
lafhterlearn word_distr word_distribution.csv --corpora-dir path/to/corpora --with-freq
```

### Word samplers
A data generator that creates synthetic word images uses a word sampler.
A sampler is a callable that returns a random word on every call. There are two different sampler classes.
The first and simplest one is UniformSampler.
With this sampler, words are taken from a uniform distribution over all dictionary words so 
that every word will appear with equal probability.

The other is FrequencyBasedSampler. As its name suggests, it samples words according to 
their probabilities/frequencies. Words with higher probability will appear more frequently. 

You can specify which one you need in the configuration file (see the section later).

It is best to compute word frequencies on text corpora written in the style
close to the text that you expect to see in a production environment.

## 3. Real unlabeled handwriting images

The toolkit will use this data to calibrate the pre-trained neural net. Create a directory
named "tuning_data". Within it, create a subdirectory called "unlabeled" and put handwriting image
files (jpg or png) here. If you want to evaluate the fine-tuned model on realistic handwriting images, 
you can add another subdirectory called "labeled".
It should contain image files whose names (excluding the file extension part) match their transcriptions. For example,
"apple.jpg".

## 4.Configuration

The final piece is a configuration. By default, the toolkit will use default configuration options 
defined in the file default_config.yml. You can always override defaults with your configuration file.
For example, you can create a "my_config.yml" configuration file and change the value for batch size:
```
batch_size: 16
```

For more details, see the ```default_config.yml``` file containing all configuration options.

# Creating a new training session

To create a fresh new training session using a default configuration, run the following command:
```
lafhterlearn make_session
```

To create a new training session with custom configuration, add a parameter --config_file:
```
lafhterlearn make_session --config_file my_config.yml
```

This command will create a new directory called "session" in the current working directory. "session" will
contain a copy of the configuration, information about model architecture, a CSV file with training history,
and model checkpoints.

# Resume pretraining

To start/resume pretraining, run the command:
```
lafhterlearn pretrain session
```

It takes a single argument, the path to the session directory.
This command will perform pretraining of the neural net on synthetic images rendered using pseudo-handwritten fonts.
The script will save weights in a checkpoint after every epoch. It will also compute metrics and 
append them to the CSV history file. If the command gets interrupted and restarted,
it will continue from the last saved checkpoint.

# Evaluate metrics

To evaluate metrics, run the command:
```
lafhterlearn evaluate session
```

This command will load the model from the latest checkpoint.
Again, it expects the user to provide a path to the session directory.

# Fine-tuning
```
lafhterlearn finetune session
```

# Use the model to transcribe images
Finally, once you have a trained model, you can transcribe handwritten words:
```
lafhterlearn transcribe session path/to/some/image.png
```

# CLI usage

Show all available commands:
```
lafhterlearn --help
```

Show usage for a particular command (for example, data):
```
lafhterlearn data --help
```

# Support

If you find this repository useful, consider starring it by clicking at the ★ button. It will be much appreciated.
