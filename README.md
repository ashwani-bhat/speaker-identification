# Speaker Identification

### Dataset
The dataset for this project is taken from kaggle: https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset

50 speakers audio data with length more than 1 hour for each. Further, data converted to wav format, 16KHz, mono channel and is split into 1min chunks. This dataset can be used for speaker recognition kind of problems. This dataset was scraped from YouTube and Librivox.

### Code

`siarec` folder contains all the training code.

Used Siamese Network to solve this problem. 

Currenly using only 3 speaker audio spectrogram images and training a Siamese Network using Contrastive loss. 

For testing, it uses the weights of the model and predicts the output label of a speaker for a single input.

`preprocess.ipynb` is a notebook for converting audio (.wav) files to spectrogram images and saving them. 
Used only 3 speaker information (out of 50) for training our model.


### Example code run
```bash
python main.py --epoch=3 --batch_size=16 --learning_rate=0.01 --model='path-to-trained-model'
```
`--model` argument when set will perform testing on the test set.

For training, don't use `--model` argument.
