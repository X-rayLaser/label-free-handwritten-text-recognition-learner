# Introduction

The toolkit provides a set of tools for training a neural net recognize handwritten words.
No labeled data is needed. This is achieved by first pretraining the model on large set of 
synthetic handwriting images. After that the network is fine-tuned on a set of real unlabeled handwriting images 
through self-training technique.

One only needs to provide (pseudo) handwritten fonts, text corpora
(to learn probability distribution over set of words), and real handwriting images. Although images do not
need to be transcribed, small subset of transcribed/labeled images will be helpful for estimating metrics.

This implementation is based on the [paper](https://arxiv.org/abs/2206.03149).

# Prerequisites

Python version 3.8.9 or higher.

# Quick start

- Under the repository folder create a directory called "fonts" and fill it with TrueType fonts with extensions
otf or ttf
- Similarly, create a directory called "corpora"
- Download a few books from [Project Gutenberg](https://www.gutenberg.org/browse/scores/top) and save them to
"corpora" directory (alternatively, find other corpora in plain text form and put them here)
- Build word distribution and save it in "word_distribution.csv" file with this command:
```
python prepare_dictionary.py word_distribution.csv --corpora-dir corpora --with-freq
```
- Create a directory with name "tuning_data".
Within that directory create a subdirectory named "unlabeled" and put handwriting image
files (jpg or png) in there. Optionally, to evaluate your fine-tuned model, you can add another subdirectory called "labeled".
It should contain image files whose names (excluding the file extension part) match their transcriptions. For example,
an image could have this name "apple.jpg" (it will be assumed to have label "apple").
- create a configuration file called my_config.py, define a class inheriting a 
hwr_self_train.configuration.Configuration class and define some settings (these will override defaults
specified by hwr_self_train.configuration.Configuration class):
```
from hwr_self_train import configuration

class Configuration(configuration.Configuration):
    def __init__(self):
        super().__init__()
        self.image_height = 96
        self.hidden_size = 128
        self.batch_size = 32
```
- create a training session:
```
python create_session.py my_config.py
```
- Start/resume pretraining on synthetic handwriting images:
```
python pretrain.py session
```
- Fine-tune the model on real handwriting images (possibly unlabeled)
```
python finetune.py session
```
- Transcribe image with real handwritten word:
```
python transcribe.py session path/to/some/image.png
```
- Evaluate the model and compute metrics:
```
python evaluate.py session
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
python prepare_fonts.py path/to/google_fonts/repository fonts --num-fonts 250 --lang latin
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
python prepare_dictionary.py word_distribution.csv --corpora-dir path/to/corpora --with-freq
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

Final piece is a configuration class. 
You can use default configuration class if it suits your needs at ```hwr_self_train.configuration.Configuration```.
Alternatively, you can create your own class inheriting from default one and override its settings. To do so, 
create a new Python module "my_config.py". Within it, create a class and fill it with the following
(replace provided values with yours):
```
from hwr_self_train import configuration

class Configuration(configuration.Configuration):
    def __init__(self):
        super().__init__()
        
        # specifies the batch size used for both pretraining and fine-tuning phases
        self.batch_size = 32
        
        # relative path to a directory of fonts 
        self.fonts_dir = './fonts'
        
        # set the word sampler to be based on word frequencies/probabilities
        self.word_sampler = 'hwr_self_train.word_samplers.FrequencyBasedSampler'
        
        # relative path to the word probabilities 
        self.sampler_data_file = "word_distribution.csv"
        
        # relative path to the directory with real data used to fine-tune the model on
        self.tuning_data_dir = 'tuning_data'
        
        # more options to override
```

For more details, see ```hwr_self_train.configuration.Configuration``` class to see which configuration options
are possible.

# Creating a new training session

To create a fresh new training session using a default configuration, run the following command:
```
python create_session.py hwr_self_train.configuration
```

This command will create a new directory called "session" in the current working directory. "session" will
contain copy of configuration, information about model architecture, CSV with training history,
and model checkpoints.

# Resume pretraining

To start/resume pretraining, run the command:
```
python pretrain.py session
```

It takes a single argument, the path to the session directory.
This command will perform pretraining the neural net on synthetic images rendered using pseudo handwritten fonts.
After every epoch, weights will be saved in a checkpoint. Metrics computed after every epoch will also be appended
to the CSV file keeping history of metrics. If command gets interrupted and restarted,
it will continue from the last saved checkpoint.

# Evaluate metrics

To evaluate metrics, run the command:
```
python evaluate.py session
```

In order to evaluate the model it will be loaded from the latest checkpoint.
Again, it expects user to provide a path to the session directory

# Fine-tuning
```
python finetune.py session
```

# Use the model to transcribe images
Finally, once you have a trained model, you can transcribe real handwritten words:
```
python transcribe.py session path/to/some/image.png
```
