import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np
import pickle

import sklearn.metrics as metrics
import os
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy import spatial
import sklearn.metrics as Metrics

import itertools
from collections import Counter
import time
import seaborn as sns

import matplotlib.pyplot as plt

class Options(object):
    def __init__(self):
        # GPU id
        self.gpu_id = 0
        ###################### Data ##########################
        # Random state of spliting data
        self.rs = None
        # If include dcodes
        self.dcode = False
        # Number of tests
        self.num_test = None
        # Current dataset number
        self.num_cur = 0
        # Maximum number of words in a review
        self.maxlen = None
        # Maximum number of notes for one patient
        self.maxnotes = None
        # Number of unique words in all reviews
        self.uniq_words = None
        # Number of training samples at validation step
        self.samples = None
        #################### Embeddings #########################
        # Vector size for each word embeddings from GloVe
        self.emb_size = 300
        # word vectors
        self.W_emb = None
        # class vectors
        self.W_class_emb = None
        # Number of classes
        self.num_class = None
        # class names
        self.class_name = None
        # ngram
        self.ngram = 20
        ###################### Model #########################
        # Training Batch Size
        self.batch_size = 20
        # Epoch
        self.epoch = 251
        # Learning rate
        self.lr_rate = 0.001
        # keep_prob, dropout_rate = 1 - keep_prob, here is the keep_prob rate
        self.keep_prob = 0.8
        # hidden units for notes
        self.H_dis = 4
        # Optimizer
        self.optimizer = 'Adam'
        # Validation Frequency
        self.valid_freq = 100
        # Early Stopping
        self.early_stop = False
        # Patience
        self.patience = None
        # Encoder
        self.encoder = "None"
        # Dilation rate
        self.l = 1
        # kernel size for tcn
        self.k = 3
        # number of filters
        self.num_filters = 8
        # save model path
        self.save_path = './save/leam_att/att_'


def leam(x_emb, x_mask, x_mask_notes, W_class_1, opt, is_training, W_class_2=None):
    """ Attention embedding encoder for hierarchical LEAM structure

    Args:
        x_emb: embedding vectors for one batch
        x_mask: x_mask matrix for x_emb
        x_mask_notes: x_mask for notes
        W_class_tran: transpose of label embeddings
        opt: option class

    Return:
        H_enc: label-based attention score encoder b * e
    """
    print("--------------------- Encoding LEAM-hier ----------------------")
    x_emb_ = tf.cast(x_emb, tf.float32)  # b * m * s * e
    x_mask_ = tf.expand_dims(x_mask, -1)  # b * m * s * 1
    x_mask_ = tf.cast(x_mask_, tf.float32)
    x_mask_notes_ = tf.expand_dims(x_mask_notes, -1)  # b * m * 1 * 1
    x_mask_notes_ = tf.cast(x_mask_notes_, tf.float32)
    x_emb_1 = tf.multiply(x_emb_, x_mask_)  # b * m * s * e
    x_emb_norm = tf.nn.l2_normalize(x_emb_1, axis=-1)
    W_class_1 = tf.cast(W_class_1, tf.float32)
    W_class_norm_1 = tf.nn.l2_normalize(W_class_1, axis=0)
    W_class_norm_1 = tf.cast(W_class_norm_1, tf.float32)
    G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm_1)  # b * m * s * c
    #     print("shape of cos similarity for emb and class: ", G.shape)
    u_conv = tf.layers.conv2d(G, filters=2, kernel_size=[1, opt.ngram], padding="same", activation=tf.nn.relu)
    att_v = tf.reduce_max(u_conv, axis=-1, keepdims=True)
    #     print("shape of maxpooling layer: ", att_v.shape)
    att_v_max = partial_softmax(att_v, x_mask_, 2, "Att_v_max", weight_notes=x_mask_notes_)
    #     print("shape of partial softmax: ", att_v_max.shape)
    x_att = tf.multiply(x_emb_, att_v_max)
    #     print("multiply attention to embeddings: ", x_att.shape)
    z = tf.reduce_sum(x_att, axis=2)
    print("shape of aggregated attentive embeddings: ", z.shape)
    H_enc = z
    #     print("shape of weighted note embeddings: ", z_weighted.shape)
    #     H_enc = tf.reduce_max(z_weighted, axis=1)
    print("shape of LEAM encoder: ", H_enc.shape)
    print("----------------------- End of Encoding --------------------------")
    return H_enc

def temporal_block(x, x_mask_notes, dropout, opt, is_training):
    print("---- dialation {0} ----".format(opt.l))
    padding = (opt.k - 1) * opt.l
    # masked note embeddings
    x_masked_notes = tf.multiply(x, x_mask_notes)
    x_padded = tf.pad(x_masked_notes, tf.constant([(0, 0), (padding, 0), (0, 0)]))
    # 1st tcn layer with dialation rate l and kernel size k
    tcn_1 = tf.layers.conv1d(x_padded, filters=opt.num_filters, kernel_size=opt.k, padding='valid',
                             dilation_rate=opt.l, activation=tf.nn.relu)
    tcn_1_norm = tf.contrib.layers.layer_norm(tcn_1)
    tcn_1_output = tf.layers.dropout(tcn_1_norm, rate=dropout, training=is_training, noise_shape = [1,1,opt.num_filters])
    # print(tcn_1_output.shape)
    # 2nd tcn layer with same specs
    tcn_1_output_masked = tf.multiply(tcn_1_output, x_mask_notes)
    x_padded_2 = tf.pad(tcn_1_output_masked, tf.constant([(0, 0), (padding, 0), (0, 0)]))
    tcn_2 = tf.layers.conv1d(x_padded_2, filters=opt.num_filters, kernel_size=opt.k, padding='valid',
                             dilation_rate=opt.l, activation=tf.nn.relu)
    tcn_2_norm = tf.contrib.layers.layer_norm(tcn_2)
    tcn_2_output = tf.layers.dropout(tcn_2_norm, rate=dropout, training=is_training,
                                    noise_shape = [1,1,opt.num_filters])
    print(tcn_2_output.shape)
    return tcn_2_output


def emb_classifier(x, x_mask, x_mask_notes, y, dropout, opt, is_training):
    x_emb, W_norm = embedding(x, opt)  # b * m * s * e
    #     print("Embedding shape: ", x_emb.shape)
    y_pos = tf.argmax(y, -1)
    y_emb_1, W_class_1 = embedding_class(y_pos, opt, 'class_emb')  # b * e, c * e
    #     print("-shape of class embedding: ", y_emb.shape)
    W_class_tran_1 = tf.transpose(W_class_1, [1, 0])  # e * c
    H_enc = leam(x_emb, x_mask, x_mask_notes, W_class_tran_1, opt, is_training)
    # first block
    layer_1 = temporal_block(H_enc, x_mask_notes, dropout, opt, is_training)
    # second block
    opt.l = 2
    layer_2 = temporal_block(layer_1, x_mask_notes, dropout, opt, is_training)
    # third block
    opt.l = 4
    layer_3 = temporal_block(layer_2, x_mask_notes, dropout, opt, is_training)
    # fourth block
    opt.l = 8
    layer_4 = temporal_block(layer_3, x_mask_notes, dropout, opt, is_training)
    # print(layer_3.shape)
    # print(layer_3[:, -1, :].shape)
    H_enc_fin = layer_4[:, -1, :]

    #     logits = discriminator_2layer(z_fin, opt, dropout, is_training)
    logits = discriminator_2layer(H_enc_fin, opt, dropout, is_training)
    #     logits = tf.layers.dense(H_enc_fin, 1, activation=None, kernel_initializer=tf.orthogonal_initializer())
    prob = tf.nn.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    saver = tf.train.Saver()

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = tf.train.AdamOptimizer(opt.lr_rate).minimize(loss)
    return prob, loss, train_step, H_enc_fin, W_norm, W_class_1, saver, layer_3