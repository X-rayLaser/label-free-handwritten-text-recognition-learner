# Label-Free Handwritten Text Recognition Learner 

It is a toolkit providing 
a set of tools for training a neural net to recognize handwritten words.
No labeled data (i.e. image transcripts) is needed. This is achieved by first pretraining the model on large set of 
synthetic handwriting images. After that the network is fine-tuned on a set of real unlabeled handwriting images 
through self-training technique.

This implementation is based on the following paper: [Self-Training of Handwritten Word Recognition for
Synthetic-to-Real Adaptation](https://arxiv.org/abs/2206.03149).

# What is taken care of

Concretely, the toolkit implements the following tasks:
- extracting probability distribution from text corpora
- generating synthetic handwritten word images
- training on synthetic images
- fine-tuning on real (handwritten word) images
- evaluating a pretrained model by computing metrics (loss, CER, WER, etc.)
- using the pretrained model to recognize handwritten words
- saving/retrieving model checkpoints 

# What needs to be provided
- handwritten fonts
- text corpora
- real unlabeled images of handwritten words
- optionally, real labeled images of handwritten words (see the note)

Note: although images do not
need to be transcribed, small subset of transcribed/labeled images will be helpful for estimating metrics.

# Prerequisites

Python version 3.8.9 or higher.

# Installation

It is recommended to install the package using virtual environment, rather than installing it globally.
The easiest way is to install it from PyPI:
```
pip install lafhterlearn
```

Alternatively, you can clone the repository, then within the repository run:
```
pip install -r requirements.txt
```

# Quick start

- Under the repository folder create a directory called "fonts" and fill it with TrueType fonts with extensions
otf or ttf
- Similarly, create a directory called "corpora"
- find text corpora and store them in the "corpora" directory as plain text files 
(you can find a few books to download [here](https://www.gutenberg.org/browse/scores/top))
- Build word distribution and save it in "word_distribution.csv" file with this command:
```
lafhterlearn word_distr word_distribution.csv --corpora-dir corpora --with-freq
```
- Create a directory with name "tuning_data".
Within that directory create a subdirectory named "unlabeled" and put handwriting image
files (jpg or png) in there. Optionally, to evaluate your fine-tuned model, you can add another subdirectory called "labeled".
It should contain image files whose names (excluding the file extension part) match their transcriptions. For example,
an image could have this name "apple.jpg" (it will be assumed to have label "apple").
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
- Fine-tune the model on real handwriting images (possibly unlabeled)
```
lafhterlearn finetune session
```
- Transcribe image with real handwritten word:
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
Training consists of 2 separate stages.

During the first stage, the net is pretrained on synthetic images
of different words rendered using random pseudo-handwritten fonts.

During the second stage, 
the net pretrained in the previous stage is fine-tuned on a (unlabeled) dataset of real handwriting
images through a procedure of self-training.
This procedure consists of repeating the following sequence of steps:
- recognize unlabeled images
- keep only confidently recognized images
- create a dataset from these recognized images
- train the neural net on that dataset

# Preparations

Before proceeding to training, we need to perform a few steps.

## 1. Getting fonts

In order to pretrain the net on synthetic images of (pseudo) handwritten words, we need to render them using
a large variety of handwritten fonts. The repository already comes with built-in image generator that generates
(image, transcript) pairs. All that is left is to fetch some fonts to use it.

Google fonts is a big repository containing a lot of different fonts including handwritten ones.
You can get all of them by cloning their repository from Github:
```
git clone https://github.com/google/fonts.git
```

Now, execute the following command to extract only handwritten fonts.
In this case it will extract up to 250 handwritten fonts with latin charset and save them to "fonts" directory:
```
lafhterlearn extract_fonts path/to/google_fonts/repository fonts --num-fonts 250 --lang latin
```

Alternatively, you can manually create fonts directory and fill it with fonts of your choosing.

## 2. Dictionary and word distribution

Another thing that we need for image generator is either a dictionary of words or word distribution table.
Dictionary should be a plain text file consisting of multiple lines, one word per line.
Word distribution should be a file in CSV format in which each line consists of a word and its
probability. The next optional section gives a little more details.

If you have a directory containing text corpora as a plain text files,
you can build word distribution with this command:
```
lafhterlearn word_distr word_distribution.csv --corpora-dir path/to/corpora --with-freq
```

### Word samplers
Data generator that creates synthetic word images makes use of a word sampler.
Sampler is simply a callable that returns a random word on every call. There are 2 different sampler classes.
The first and the simplest one is UniformSampler.
With this sampler, words are taken from uniform distribution over all dictionary words so that every word in it 
is equally likely to appear.

The other is FrequencyBasedSampler. As its name suggests, it samples words according to their probabilities/frequencies.
Words with higher probability will appear more frequently. 

You can specify which one you need in the configuration file (see section later).

It is best to compute word frequencies on text corpora written in the style
close to text that you expect to see in production environment.

## 3. Real unlabeled handwriting images

This is the data that will be used to calibrate pretrained neural net. Simply create a directory
with name "tuning_data". Within that directory create a subdirectory named "unlabeled" and put handwriting image
files (jpg or png) in there. Optionally, to evaluate your fine-tuned model, you can add another subdirectory called "labeled".
It should contain image files whose names (excluding the file extension part) match their transcriptions. For example,
"apple.jpg".

## 4.Configuration

Final piece is a configuration. By default, default configuration options are used.
For documentation purpose, these are defined in file default_config.yml under the root of the repository.
You can always override defaults with your own configuration file.
For example, you can create "my_config.yml" configuration file and change the value for batch size:
```
batch_size: 16
```

For more details, see ```default_config.yml``` file to see which configuration options
are possible.

# Creating a new training session

To create a fresh new training session using a default configuration, run the following command:
```
lafhterlearn make_session
```

To create new training session with custom configuration, add a parameter --config_file:
```
lafhterlearn make_session --config_file my_config.yml
```

This command will create a new directory called "session" in the current working directory. "session" will
contain copy of configuration, information about model architecture, CSV file with training history,
and model checkpoints.

# Resume pretraining

To start/resume pretraining, run the command:
```
lafhterlearn pretrain session
```

It takes a single argument, the path to the session directory.
This command will perform pretraining the neural net on synthetic images rendered using pseudo handwritten fonts.
After every epoch, weights will be saved in a checkpoint. Metrics computed after every epoch will also be appended
to the CSV file keeping history of metrics. If command gets interrupted and restarted,
it will continue from the last saved checkpoint.

# Evaluate metrics

To evaluate metrics, run the command:
```
lafhterlearn evaluate session
```

In order to evaluate the model it will be loaded from the latest checkpoint.
Again, it expects user to provide a path to the session directory

# Fine-tuning
```
lafhterlearn finetune session
```

# Use the model to transcribe images
Finally, once you have a trained model, you can transcribe real handwritten words:
```
lafhterlearn transcribe session path/to/some/image.png
```
