# LanguageClassifier
Model to classify the language of a speaker given a wav file. 
This is an adaptation of the paper Exploring Wav2vec2.0 on Speaker Identification and Language Verification (Zhiyun Fan et al in Computing Research Repository, CoRR, 2020).
I fine-tune the multi-lingual Wav2Vec2.0 model (XLSR Large) on a Language Identification task. Initially I only classify English, Swedish and Arabic but this is extendable). XLSR Large was trained on 53 languages from the LibriSpeech, Commonvoice and Babel datasets and is available at Huggingface Transformers.
The model is trained on the full Commonvoice dataset for the English, Swedish and Arabic languages, a total of 3000 samples, of which 10% are held-back test set and 5% used for validation. This results in a total of 2561 training samples.

This is the full architecture:
An encoder consisting of the Wav2Vec multilingual model (from transformers import Wav2VecForCTC) with a Language Modeling head, followed by an average pooling layer which takes a simple mean of the output from the language model (modeloutput.logits) and a fully connected layer that classifies into one of X (at the moment 3) languages.

*Training the Model:*
python train.py --dataset_dir=<specify path to training data directory> --checkpoints_dir=<specify folder where you want checkpoints saved (will create if it does not exist)

For full options type python train.py -h help

*Testing the model*
python test.py --model_checkpoint= <specify path to model checkpoint to test> --test_data_dir=<specify path to test data directory>

The best performing checkpoints will be uploaded to GitHub.
