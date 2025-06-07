import numpy as np
import pandas as pd

import torch
import transformers as ppb
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

import os
import subprocess

os.makedirs('/content/module_useT', exist_ok=True)
subprocess.run(
    'curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /content/module_useT',
    shell=True,
    check=True
)

embed = hub.Module("/content/module_useT")

#GUSE transformer
def encodeData(messages): 
  # Red logging output.
  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(embed(messages))

  final_embeddings = pd.DataFrame(data=message_embeddings)

  return final_embeddings

# text feat -> word embedding
training_regular = pd.read_csv('../data/training-set.csv')['selftext']
new_training_regular = encodeData(training_regular)
new_training_regular.to_csv('guse-training-features.csv')


model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def getFeatures(batch_1):

  tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)))

  max_len = 0
  for i in tokenized.values:
      if len(i) > max_len:
          max_len = len(i)

  padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


  attention_mask = np.where(padded != 0, 1, 0)
  attention_mask.shape


  input_ids = torch.tensor(padded)  
  attention_mask = torch.tensor(attention_mask)

  with torch.no_grad():
      last_hidden_states = model(input_ids, attention_mask=attention_mask)

  features = last_hidden_states[0][:,0,:].numpy() #2d bert feats
  # features = last_hidden_states[0].numpy() #3d bert feats 

  return features

df = pd.read_csv('real-training-set.csv', delimiter=',')
df = df[['selftext', 'is_suicide']]
df = df.rename(columns={'selftext': 0, 'is_suicide': 1})

bert_features = getFeatures(df)
np.savetxt("bert-training-features.csv", bert_features, delimiter=',')