# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bert import modeling
import tokenization_ner
import tensorflow as tf
import numpy as np
from tensorflow.contrib import crf
from tensorflow.contrib.layers.python.layers import initializers

DTYPE = tf.float32
DTYPE_INT = tf.int32


class charBERT(object):
    def __init__(self,
               bert_config, char_config,
               is_training, # is_evaluation,
               input_token_ids, input_char_ids,
               labels, num_labels, use_char_representation=True,
               input_mask=None, segment_ids=None,
               use_one_hot_embeddings=False, # TPU加速则为True
               scope=None):
        """

        :param bert_config:
        :param char_config:
        :param is_training: 处于estimator模式下的train模式
        :param is_evaluation: 处于estimator模式下的evaluate模式
        :param input_token_ids:
        :param input_char_ids:
        :param labels: 真实标签
        :param num_labels: 标签个数，用于CRF的转移矩阵
        :param input_mask:
        :param segment_ids: 用于Bert，不过这里没啥用处，因为只是处理一个ner的问题，所以bert默认都为0
        :param use_one_hot_embeddings: 是否用tpu
        :param scope:
        """
        self.bert_model = modeling.BertModel(config=bert_config,
                                        is_training=is_training,
                                        input_ids=input_token_ids,
                                        input_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        use_one_hot_embeddings=use_one_hot_embeddings)
        self.token_output = self.bert_model.get_sequence_output()

        if use_char_representation:
            char_embed_dim = char_config['char_embed_dim']
            filters = char_config['filters']
            alphabet_size = char_config['alphabet_size']
            activations = char_config['activations']
            n_highway = char_config['n_highway']
            projection_dim = char_config['projection_dim']
            char_dropout_rate = char_config['char_dropout_rate'] if is_training else 1.0

            self.charcnn_model = CharRepresentation(char_input=input_char_ids,
                                         alphabet_size=alphabet_size,
                                         filters=filters,
                                         projection_dim=projection_dim,
                                         char_embed_dim=char_embed_dim,
                                         activations=activations,
                                         n_highway=n_highway,
                                         dropout_rate=char_dropout_rate
                                        )
            self.char_output = self.charcnn_model.get_highway_output()

            token_shape = modeling.get_shape_list(self.token_output, expected_rank=3)
            char_shape = modeling.get_shape_list(self.char_output, expected_rank=3)

            if token_shape[1] != char_shape[1]:
                raise ValueError(
                    "The time steps of token representation (%d) is not the same as char representation (%d) "
                     % (token_shape[1], char_shape[1]))

            self.final_output = tf.concat([self.token_output, self.char_output], axis=-1)
        else:
            tf.logging.info("****************BERT representation only***************")
            self.final_output = self.token_output

        sequece_lengths = tf.reduce_sum(input_mask, axis=-1)
        self.crf = CRF(input=self.final_output,
                       labels=labels,
                       num_labels=num_labels,
                       lengths=sequece_lengths,
                       is_training=is_training,
                       # is_evaluation=is_evaluation  # estimator模式下的evaluate模式还是需要返回损失函数的
                       )

    def get_crf_loss(self):
        return self.crf.crf_loss()

    def get_orig_loss(self):
        return self.crf.orig_loss()

    def get_crf_preds(self):
        return self.crf.get_crf_decode_tags()

    def get_orig_preds(self):
        return self.crf.get_orig_tags()


class CRF(object):
    def __init__(self, input, labels, num_labels,
                lengths, is_training, dropout_rate=0.7):
        """

        :param input:
        :param labels:
        :param num_labels: label的种类数，因为CRF是状态转移，因此label为一个状态
        :param lengths: batch中每个句子的实际长度
        :param is_training:
        :param dropout_rate:
        """
        self.labels = labels
        self.num_labels = num_labels

        if is_training:
            input = tf.nn.dropout(input, dropout_rate)
        # project
        self.logits = self._project_layer(input, num_labels)
        if is_training:
            self.logits = tf.nn.dropout(self.logits, dropout_rate)
        # crf
        self.log_likelihood, self.trans = self._crf_log_likelihood(self.labels, self.logits, lengths, num_labels)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        self.pred_ids, _ = crf.crf_decode(potentials=self.logits, transition_params=self.trans, sequence_length=lengths)

    def _project_layer(self, input, num_labels, name=None):
        """
        :param outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        hidden_state = input.get_shape()[-1]
        seq_length = input.get_shape()[-2]
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_state, num_labels],
                                    dtype=tf.float32, initializer=initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                hidden_ouput = tf.reshape(input,[-1, hidden_state])
                pred = tf.nn.xw_plus_b(hidden_ouput, W, b)
            return tf.reshape(pred, [-1, seq_length, num_labels])

    def _crf_log_likelihood(self, labels, logits, lengths, num_labels):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[num_labels, num_labels],
                initializer=initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=lengths)
            # return tf.reduce_mean(-log_likelihood), trans
            return log_likelihood, trans

    def crf_loss(self):
        return tf.reduce_mean(-self.log_likelihood)

    def orig_loss(self):
        self.labels = tf.one_hot(indices = self.labels, depth = self.num_labels)
        self.loss_per_loc = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,
                                                                       logits=self.logits,
                                                                       dim=-1)
        return tf.reduce_mean(tf.reduce_sum(self.loss_per_loc, axis=-1), # 每个example的每个位置tag的损失，再加起来
                            axis=-1)

    def get_crf_decode_tags(self):
        return self.pred_ids

    def get_orig_tags(self):
        return tf.argmax(self.logits, axis=-1)


class CharRepresentation(object):
    def __init__(self, char_input, alphabet_size, filters,
                 char_embed_dim, projection_dim, activations='tanh',
                 n_highway=None, dropout_rate=0.7):

        char_length = char_input.get_shape().as_list()[2]
        sequence_length = char_input.get_shape().as_list()[1]
        batch_size = char_input.get_shape().as_list()[0]

        with tf.name_scope("Char_Embedding"), tf.device('/cpu:0'):
            self.embedding_weights = tf.get_variable(  # 为每个字符形成的嵌入表
                "char_embed", [alphabet_size, char_embed_dim], dtype=DTYPE,
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         char_input)

        # for first model, this is False, for others it's True
        n_filters = sum(f[1] for f in filters)
        reuse = tf.get_variable_scope().reuse
        self.sequence_output = add_char_convolution(self.char_embedding, filters, activations, reuse)

        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim
        if use_highway or use_proj:
            self.sequence_output = tf.reshape(self.sequence_output, [-1, n_filters])

        if use_highway:
            self.sequence_output = highway(self.sequence_output, n_highway)

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable(
                    "W_proj", [n_filters, projection_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                    dtype=DTYPE)
                b_proj_cnn = tf.get_variable(
                    "b_proj", [projection_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=DTYPE)
                self.sequence_output = tf.matmul(self.sequence_output, W_proj_cnn) + b_proj_cnn

        if use_highway or use_proj:
            orig_shape = [-1, sequence_length, projection_dim]
            self.sequence_output = tf.reshape(self.sequence_output, orig_shape)
            self.sequence_output = tf.nn.dropout(self.sequence_output, dropout_rate)

    def get_embedding_output(self):
        return self.char_embedding

    def get_highway_output(self):
        return self.sequence_output


def add_char_convolution(input, filters, activations, reuse):
    # input shape (batch_size, unroll_steps, max_chars, embed_dim)
    char_embed_dim = input.get_shape().as_list()[-1]
    char_length = input.get_shape().as_list()[-2]
    with tf.variable_scope("CNN", reuse=reuse):
        convolutions = []
        for i, (width, num_filters) in enumerate(filters):
            if activations == 'relu':
                # He initialization for ReLU activation
                # with char embeddings init between -1 and 1
                # w_init = tf.random_normal_initializer(
                #    mean=0.0,
                #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                # )

                # Kim et al 2015, +/- 0.05
                w_init = tf.random_uniform_initializer(
                    minval=-0.05, maxval=0.05)
                activation = tf.nn.relu
            elif activations == 'tanh':
                # glorot init
                w_init = tf.random_normal_initializer(
                    mean=0.0,
                    stddev=np.sqrt(1.0 / (width * char_embed_dim))
                )
                activation = tf.nn.tanh
            w = tf.get_variable(  # 一个一维的卷积
                "W_cnn_%s" % i,
                # height, width, in_channel, out_channel, 这里的height设为1，因为只考虑一个单词内的字母排列，width为每次考虑width个字母
                # 后续卷积后的shape为(batch_size, sequence_length, char_length - width + 1, num_filters)
                [1, width, char_embed_dim, num_filters],
                initializer=w_init,
                dtype=DTYPE)
            b = tf.get_variable(  # out_channel
                "b_cnn_%s" % i, [num_filters], dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(  # 卷积，从左到右
                input, w,
                strides=[1, 1, 1, 1],
                padding="VALID") + b
            # now max pool
            # 使用一个max pool，每一行进行pooling
            # 取这些字母卷积后，最耀眼的一个位置，因此max_pool以后shape为(batch_size, sequence_length, 1, num_filters)
            # 这里可否把max_pool换成一个层叠卷积呢？
            conv = tf.nn.max_pool(
                conv, [1, 1, char_length - width + 1, 1],
                [1, 1, 1, 1], 'VALID')

            # activation
            conv = activation(conv)
            conv = tf.squeeze(conv, squeeze_dims=[2])

            convolutions.append(conv)

        return tf.concat(convolutions, 2)

def add_char_recurrent(input, filters, activations, reuse, bidirectional):
    pass


# 参考highway网络的定义
def highway(input, n_highway):
    highway_dim = input.get_shape().as_list()[-1]
    sequence_length = input.get_shape().as_list()[-2]
    for i in range(n_highway):
        with tf.variable_scope('high_%s' % i) as scope:
            W_carry = tf.get_variable(  # 这些都是get_variable
                'W_carry', [highway_dim, highway_dim],
                # glorit init
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                dtype=DTYPE)
            b_carry = tf.get_variable(
                'b_carry', [highway_dim],
                initializer=tf.constant_initializer(-2.0),
                dtype=DTYPE)
            W_transform = tf.get_variable(
                'W_transform', [highway_dim, highway_dim],
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                dtype=DTYPE)
            b_transform = tf.get_variable(
                'b_transform', [highway_dim],
                initializer=tf.constant_initializer(0.0),
                dtype=DTYPE)
    input = tf.reshape(input, [-1, highway_dim])

    carry = tf.matmul(input, W_carry) + b_carry
    carry_gate = tf.nn.sigmoid(carry)

    transform = tf.matmul(input, W_transform) + b_transform
    transform_gate = tf.nn.relu(transform)

    return carry_gate * transform_gate + (1.0 - carry_gate) * input





