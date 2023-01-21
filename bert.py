import pandas as pd
from transformers import TFBertModel, BertTokenizer
seed_value = 29
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
np.set_printoptions(precision=2)
import tensorflow as tf
tf.random.set_seed(seed_value)
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

N_AXIS = 4
MAX_SEQ_LEN = 128
BERT_NAME = 'bert-base-uncased'
'''
EMOTIONAL AXES:
Introversion (I) – Extroversion (E)
Intuition (N) – Sensing (S)
Thinking (T) – Feeling (F)
Judging (J) – Perceiving (P)
'''
axes = ["I-E","N-S","T-F","J-P"]
classes = {"I":0, "E":1, # axis 1
           "N":0,"S":1, # axis 2
           "T":0, "F":1, # axis 3
           "J":0,"P":1} # axis 4

def text_preprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]
    return text

train_n=6624
val_n=1024
test_n=1024
import csv

# with open("/Users/himayaperera/Desktop/4thyear/fyp/bert/mb.csv", 'r') as file:
#   csvreader = csv.reader(file)
#   for row in csvreader:
#     print(row)
# data = pd.read_csv("/Users/himayaperera/Desktop/4thyear/fyp/bert/mb.csv")
# data = data.sample(frac=1)
# labels = []
# print(data)
# for personality in data["type"]:
#     pers_vect = []
#     for p in personality:
#         pers_vect.append(classes[p])
#     labels.append(pers_vect)
# sentences = data["posts"].apply(str).apply(lambda x: text_preprocessing(x))
# print("step2:")
# print("step2:",sentences)
# labels = np.array(labels, dtype="float32")
# train_sentences = sentences[:train_n]
# y_train = labels[:train_n]
# val_sentences = sentences[train_n:train_n+val_n]
# y_val = labels[train_n:train_n+val_n]
# test_sentences = sentences[train_n+val_n:train_n+val_n+test_n]
# y_test = labels[train_n+val_n:train_n+val_n+test_n]

# need this
def prepare_bert_input(sentences, seq_len, bert_name):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    encodings = tokenizer(sentences.tolist(), truncation=True, padding='max_length',
                                max_length=seq_len)
    input = [np.array(encodings["input_ids"]), np.array(encodings["token_type_ids"]),
               np.array(encodings["attention_mask"])]
    return input

# X_train = prepare_bert_input(train_sentences, MAX_SEQ_LEN, BERT_NAME)
# X_val = prepare_bert_input(val_sentences, MAX_SEQ_LEN, BERT_NAME)
# X_test = prepare_bert_input(test_sentences, MAX_SEQ_LEN, BERT_NAME)

print("step 3:")
input_ids = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='input_ids')
input_type = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='token_type_ids')
input_mask = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name='attention_mask')
inputs = [input_ids, input_type, input_mask]
bert = TFBertModel.from_pretrained(BERT_NAME)
bert_outputs = bert(inputs)
last_hidden_states = bert_outputs.last_hidden_state
avg = layers.GlobalAveragePooling1D()(last_hidden_states)
output = layers.Dense(N_AXIS, activation="sigmoid")(avg)
model = keras.Model(inputs=inputs, outputs=output)
model.summary()
model.load_weights('weights.h5')

# max_epochs = 7
# batch_size = 32
# opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
# loss = keras.losses.BinaryCrossentropy()
# best_weights_file = "weights.h5"
# auc = keras.metrics.AUC(multi_label=True, curve="ROC")
# m_ckpt = ModelCheckpoint(best_weights_file, monitor='val_'+auc.name, mode='max', verbose=2,
#                           save_weights_only=True, save_best_only=True)
# model.compile(loss=loss, optimizer=opt, metrics=[auc, keras.metrics.BinaryAccuracy()])
# model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=max_epochs,
#     batch_size=batch_size,
#     callbacks=[m_ckpt],
#     verbose=2
# )

# loss = keras.losses.BinaryCrossentropy()
# best_weights_file = "weights.h5"
# model.load_weights(best_weights_file)
# opt = tfa.optimizers.RectifiedAdam(learning_rate=3e-5)
# model.compile(loss=loss, optimizer=opt, metrics=[keras.metrics.AUC(multi_label=True, curve="ROC"),
#                                                   keras.metrics.BinaryAccuracy()])
# predictions = model.predict(X_test)
# model.evaluate(X_test, y_test, batch_size=32)

# def plot_roc_auc(y_test, y_score, classes):
#     assert len(classes) > 1, "len classes must be > 1"
#     plt.figure()
#     if len(classes) > 2:  # multi-label
#         # Compute ROC curve and ROC area for each class
#         for i in range(len(classes)):
#             fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
#             roc_auc = auc(fpr, tpr)
#             plt.plot(fpr, tpr, label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc))
#         # Compute micro-average ROC curve and ROC area
#         fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
#         roc_auc = auc(fpr, tpr)
#         # Plot ROC curve
#         plt.plot(fpr, tpr, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc))
#     else:
#         fpr, tpr, _ = roc_curve(y_test, y_score)
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()

# plot_roc_auc(y_test, predictions, axes)

# import torch
# torch.save(model, 'bertmodel')
# # model.save("my_model")
# model.save_pretrained("bertbert")
# model.save_weights('bert_weights.h5')
# model.save('model')
# import joblib

# joblib.dump(model, 'bertbert')
# trainer = joblib.load('bertbert')
# from torch import nn
# from transformers import Trainer
# Trainer.save_model('model')
# trainer.model.predict(["text", "text"])
# savedmodel = torch.load('bertmodel')

s1 = "'I'm finding the lack of me in these posts very alarming.|||Sex can be boring if it's in the same position often. For example me and my girlfriend are currently in an environment where we have to creatively use cowgirl and missionary. There isn't enough...|||Giving new meaning to 'Game' theory.|||Hello *ENTP Grin*  That's all it takes. Than we converse and they do most of the flirting while I acknowledge their presence and return their words with smooth wordplay and more cheeky grins.|||This + Lack of Balance and Hand Eye Coordination.|||Real IQ test I score 127. Internet IQ tests are funny. I score 140s or higher.  Now, like the former responses of this thread I will mention that I don't believe in the IQ test. Before you banish...|||You know you're an ENTP when you vanish from a site for a year and a half, return, and find people are still commenting on your posts and liking your ideas/thoughts. You know you're an ENTP when you...|||http://img188.imageshack.us/img188/6422/6020d1f9da6944a6b71bbe6.jpg|||http://img.adultdvdtalk.com/813a0c6243814cab84c51|||I over think things sometimes. I go by the old Sherlock Holmes quote.  Perhaps, when a man has special knowledge and special powers like my  own, it rather encourages him to seek a complex...|||cheshirewolf.tumblr.com  So is I :D|||400,000+  post|||Not really; I've never thought of E/I or J/P as real functions.  I judge myself on what I use. I use Ne and Ti as my dominates. Fe for emotions and rarely Si. I also use Ni due to me strength...|||You know though. That was ingenious. After saying it I really want to try it and see what happens with me playing a first person shooter in the back while we drive around. I want to see the look on...|||out of all of them the rock paper one is the best. It makes me lol.  You guys are lucky :D I'm really high up on the tumblr system.|||So did you hear about that new first person shooter game? I've been rocking the hell out of the soundtrack on my auto sound equipment that will shake the heavens. We managed to put a couple PS3's in...|||No; The way he connected things was very Ne. Ne dominates are just as aware of their environments as Se dominates.  Example: Shawn Spencer or Patrick Jane; Both ENTPs.|||Well charlie I will be the first to admit I do get jealous like you do. I chalk it up to my 4w3 heart mixed with my dominate 7w8. 7s and 8s both like to be noticed. 4's like to be known (not the same...|||;D I'll upload the same clip with the mic away from my mouth. Than you won't hear anything.  Ninja Assassin style but with splatter.|||Tik Tok is a really great song. As long as you can mental block out the singer. I love the beat it makes me bounce.|||drop.io v1swck0  :D Mic really close to my mouth and smokin aces: assassins ball playing in the background.|||Sociable =/= extrovert; I'm an extrovert and I'm not sociable. :)|||Sherlock in the movie was an ENTP. Normally he's played as a EXTJ. In the books he's an ESTJ.  As I said. The movie looked good except for it being called sherlock holmes.|||http://i817.photobucket.com/albums/zz96/kamioo/Dirtywinch.png|||Oh, I never had fear of kissing a guy. I will kiss an animal too. So there was nothing to vanish. Just personal taste and me not liking it.  The guy I kissed didn't know me. It was one of those...|||Sounds pretty much like my area and what I'm going through right now trying to figure out which way I want to take my life. I want to do so many things. The biggest problem is that I know if I don't...|||;D I was operating under the impression that you were female. I never looked at your boxy. Okay, I help out my gay friends all the time and one of them has developed a little crush on me. I get red...|||T_T You just described me  and I'm living the worst nightmare. I'm trapped in one place with one one around. Only dull woods. If I was a serial killer this would be the perfect place but sadly I'm...|||TBH, and biased, sounds like a shadowed INFP. I think maybe he was hurt and turned ESTJ. I can tell because he has some of the typical INFP traits left over.|||*Checks list* I'm sorry. It seems that you have came at a bad time. We've already reached our quota of INFJs. However, being you're female and I like females I will make you a deal. I will kick one...|||I'm ANTP (Leaning toward E). I'm easy for both ENTPs and INTPs to identify with. :)|||I also imagine ENTP's interrogations would go a little bit like Jack's from 24 except more mechanical. Rigging up shock treatment equipment in an abandoned building out of an old car batty, jumper...|||It was a compliment :) Trust me. I'm just as psychopathic :D except I have emoticons. They're just weird ones. Like laughing when I get hurt or at people running themselves over with their lawn mower...|||http://i817.photobucket.com/albums/zz96/kamioo/Thunderstorm.pnghttp://i817.photobucket.com/albums/zz96/kamioo/Thunderstormbw.png http://i817.photobucket.com/albums/zz96/kamioo/Cosmicstorm.png|||No. It's like a theme for where I live and that is why I know it by heart.   http://www.youtube.com/watch?v=j5W73HaVQBg|||and I usual don't leave until the thing ends. But in the mean time. In between times. You work your thing. I'll work mine :D  ;D I'm the MBP; Pleasure to meet you.|||Damn, need to trust my instincts more I would have been closer I was going to say INFP.|||EXFP? Leaning toward S with the way she responded.  :D My friends, even my gay and lesbian ones, always come to me for advice.|||I bow to my entp masters ENTPs are so great. If it wasn't for ENTPs I wouldn't have been able to build what I'm building  Duck Duck  Duck  Shotgun|||What? Me? I never do that >.> <.<|||Because its hard to be sad about losing someone you like when you knew you were right and give yourself a big pat on the back because you're awesome and always correct.|||Oh, you don't have to tell me that most of them are stupid. I know this. That is why I play with them and it makes me laugh. :D As I'm going to take Neuropsychology and I have a few psychologist...|||:D I'm a Nightowl. I wake up between 6-7pm and stay awake till 10-11:30am.|||Personal opinion backed by theory would suggest that INTPs are the most socially difficult. While INTJs can be socially indifferent but they will also use social situations if the the need arises....|||Personal stocks that I have on my desktop that I've downloaded from random stock sites and stock photobuckets.|||I'll tell you when I open photoshop.  :) Glad you like it static.|||:D Thanks.|||http://i817.photobucket.com/albums/zz96/kamioo/Deathgrip.png http://i817.photobucket.com/albums/zz96/kamioo/Deathgripbw.png  Made for a friend. Several hours of work. I constructed every line by...|||:) Static: http://i817.photobucket.com/albums/zz96/kamioo/Statickitten.png  I'll have to get to your avatar later if one of my fellow teammates doesn't.|||Psychologist don't keep me around long enough to diagnosis me. I like to toy with them. What I have diagnosis myself with and had a few psychologist friends (+ a few other friends) tell me I have is...'"
sentences = np.asarray([s1])
enc_sentences = prepare_bert_input(sentences, MAX_SEQ_LEN, BERT_NAME)
# best_weights_file = "weights.h5"
# model.load_weights(best_weights_file)
predictions = model.predict(enc_sentences)
for sentence, pred in zip(sentences, predictions):
    pred_axis = []
    mask = (pred > 0.5).astype(bool)
    for i in range(len(mask)):
        if mask[i]:
            pred_axis.append(axes[i][2])
        else:
            pred_axis.append(axes[i][0])
    print('-- comment: '+sentence.replace("\n", "").strip() +
          '\n-- personality: '+str(pred_axis) +
          '\n-- scores:'+str(pred))