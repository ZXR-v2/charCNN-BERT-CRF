# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import logging
import time

import tensorflow as tf
import codecs
import tokenization_ner
import model
from bert import modeling
from bert import optimization
import tf_metrics
from ner_evaluation import Evaluator
from math import ceil
from lxml import etree

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #"-1"即为禁用GPU，只使用CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.flags

FLAGS = flags.FLAGS

bert_path = 'multi_cased_L-12_H-768_A-12'
root_path = 'E:\\毕设项目\\deid_Bert-Plus_en'

flags.DEFINE_string(
    "bert_config_file", os.path.join(root_path, bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'ner', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", os.path.join(root_path, 'output'),
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(root_path, bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_integer(
    "max_char_length", 16,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', False, 'remove the files which created by last training')

flags.DEFINE_bool("use_crf", True, "Whether to use crf decode tags.")

flags.DEFINE_bool("use_char_representation", True, "Whether to use character representation.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("pred_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5.0, "Total number of training epochs to perform.")
flags.DEFINE_float('dropout_rate', 0.7, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    def __init__(self, guid, text, tags=None):
        """Constructs a InputExample.
        Args:
          guid: the medicad record id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.tags = tags


class InputFeature(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, char_ids, tag_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.char_ids = char_ids
        self.tag_ids = tag_ids


def load_examples(data_dir):
    XMLProcessor = tokenization_ner.xmlProcessor()
    example_list = []
    for dir in data_dir:
        path = os.path.join(root_path, dir)
        files = os.listdir(path)
        for file in files:
            xml = XMLProcessor.decodeXML(path, file)
            example = InputExample(file, xml['text'], xml['tags'])
            example_list.append(example)
    return example_list


def convert_single_example(example, tokenizer):
    data = tokenizer.tokenize(example.text, example.tags, example.guid)
    features = []
    for s_id, (input_ids, input_mask, segment_ids, char_ids) in enumerate(zip(data["input_ids"], data["input_mask"],
                                                                  data["segment_ids"], data["char_ids"])):
        tag_ids = data["tag_ids"][s_id] if data["tag_ids"]!=None else None

        feature = InputFeature(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            char_ids=char_ids,
            tag_ids=tag_ids,
        )
        features.append(feature)
    return features


def filed_based_convert_examples_to_features(
        examples, tokenizer, output_file):
    """
    :param examples:
    :param tokenizer:
    :param output_file:
    :param mode:
    :return: number of small example
    """
    num_examples = 0
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        # 对于每一个训练样本,
        feature_list = convert_single_example(example, tokenizer)
        num_examples += len(feature_list)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def flatten(tensor):
            return sum(tensor, [])

        for f in feature_list:
            if num_examples%5000 == 0:
                tf.logging.info("Writing example %d of %d" % (num_examples, len(examples)))
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(f.input_ids)
            features["input_mask"] = create_int_feature(f.input_mask)
            features["segment_ids"] = create_int_feature(f.segment_ids)
            features["char_ids"] = create_int_feature(flatten(f.char_ids))
            features["tag_ids"] = create_int_feature(f.tag_ids)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("guid: %s" % (example.guid))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in f.input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in f.input_mask]))
                tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in f.segment_ids]))
                tf.logging.info("char_ids's length is : %s" % " ".join(str(len(flatten(f.char_ids)))))
                tf.logging.info("tag_ids: %s" % " ".join([str(x) for x in f.tag_ids]))

    writer.close()
    return num_examples


def file_based_input_fn_builder(input_file, seq_length, char_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "char_ids": tf.FixedLenFeature([seq_length * char_length], tf.int64),
        "tag_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        example["char_ids"] = tf.reshape(example["char_ids"],
                                         shape=(seq_length, char_length))
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


def model_fn_builder(bert_config, char_config, num_labels,
                     init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu, use_crf,
                     use_char_representation, use_one_hot_embeddings):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :param use_tpu:
    :param use_one_hot_embeddings:
    :return:
    """
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        char_ids = features["char_ids"]
        tag_ids = features["tag_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # is_evaluation = (mode == tf.estimator.ModeKeys.EVAL)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示

        charbert = model.charBERT(bert_config=bert_config,
                                  char_config=char_config,
                                  is_training=is_training,
                                  # is_evaluation=is_evaluation,
                                  use_char_representation=use_char_representation,
                                  input_token_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  input_char_ids=char_ids,
                                  labels=tag_ids,
                                  num_labels=num_labels
                                  )
        if use_crf:
            total_loss = charbert.get_crf_loss()
            pred_ids = charbert.get_crf_preds()
        else:
            total_loss = charbert.get_orig_loss()
            pred_ids = charbert.get_orig_preds()

        tvars = tf.trainable_variables()
        scaffold_fn = None
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            pos_indices = list(tokenization_ner.build_tag_vocab().values())[1:-3] # 去掉'X','BOS','EOS'作为pos_indices去预测
            def metric_fn(label_ids, pred_ids, input_mask):
                # 首先对结果进行维特比解码
                # crf 解码
                # 这里先用orig_to_tok取出有效的tag
                precision = tf_metrics.precision(labels=label_ids,
                                                 predictions=pred_ids,
                                                 num_classes=num_labels,
                                                 pos_indices=pos_indices,
                                                 weights=input_mask
                                                 )
                recall = tf_metrics.recall(labels=label_ids,
                                                 predictions=pred_ids,
                                                 num_classes=num_labels,
                                                 pos_indices=pos_indices,
                                                 weights=input_mask
                                                 )
                f = tf_metrics.f1(labels=label_ids,
                                  predictions=pred_ids,
                                  num_classes=num_labels,
                                  pos_indices=pos_indices,
                                  weights=input_mask
                                  )

                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    # "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [tag_ids, pred_ids, input_mask])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)  #
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=pred_ids,
                scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_data_dir = ['training-PHI-Gold-Set1', 'training-PHI-Gold-Set2']
    predict_data_dir = eval_data_dir = ['testing-PHI-Gold-fixed']
    tag_vocab = tokenization_ner.build_tag_vocab()

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    char_config = {}
    char_config['char_embed_dim'] = 128
    char_config['filters'] =  [                                                                     #所使用的卷积核列表
        [1, 32],                                                                     #表示n_width=1的卷积核32个
        [2, 32],
        [3, 64],
        [4, 128],
        [5, 256],
        [6, 512],
        [7, 512]
    ]
    char_config['alphabet_size'] = len(tokenization_ner.build_char_vocab())
    char_config['activations'] = 'relu'
    char_config['n_highway'] = 2
    char_config['projection_dim'] = 128
    char_config['char_dropout_rate'] = 0.7

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    config = tf.ConfigProto(allow_soft_placement=True,
                            #log_device_placement=True
                            )
    config.gpu_options.allow_growth = True

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if os.path.exists(os.path.join(FLAGS.output_dir, "data_config.json")):
        with open(os.path.join(FLAGS.output_dir, "data_config.json")) as dc:
            dc_str = dc.readline()
            data_config = json.loads(dc_str)
        dc.close()
    else:
        data_config = collections.OrderedDict()

    wordpiece_vocab = tokenization_ner.build_wordpiece_vocab(root_path, bert_path, 'vocab.txt')
    wptokenizer = tokenization_ner.WPTokenizer(wordpiece_vocab, FLAGS.max_seq_length, FLAGS.max_char_length)

    if FLAGS.do_train:
        if os.path.exists(os.path.join(FLAGS.output_dir, "train.tf_record")) \
                and "num_train_examples" in data_config.keys():
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            num_train_examples = data_config["num_train_examples"]
            num_train_steps = data_config["num_train_steps"]
            num_warmup_steps = data_config["num_warmup_steps"]
        else:
            train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
            train_examples = load_examples(train_data_dir)
            num_train_examples = filed_based_convert_examples_to_features(train_examples, wptokenizer, train_file)
            num_train_steps = ceil(
                num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
            data_config["num_train_examples"] = num_train_examples
            data_config["num_train_steps"] = num_train_steps
            data_config["num_warmup_steps"] = num_warmup_steps
            save_data_config(data_config)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num sentence examples = %d", num_train_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        char_config=char_config,
        num_labels=len(tag_vocab),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_crf=FLAGS.use_crf,
        use_char_representation=FLAGS.use_char_representation,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.pred_batch_size)

    if FLAGS.do_train:
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            char_length=FLAGS.max_char_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        if os.path.exists(os.path.join(FLAGS.output_dir, "eval.tf_record")) and \
                "num_eval_examples" in data_config.keys():
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            num_eval_examples = data_config["num_eval_examples"]
            num_eval_steps = data_config["num_eval_steps"]
        else:
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            eval_examples = load_examples(eval_data_dir)
            num_eval_examples = filed_based_convert_examples_to_features(eval_examples, wptokenizer, eval_file)
            num_eval_steps = ceil(
                num_eval_examples / FLAGS.eval_batch_size)
            data_config["num_eval_examples"] = num_eval_examples
            data_config["num_eval_steps"] = num_eval_steps
            save_data_config(data_config)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num sentence examples = %d", num_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        tf.logging.info("  Num steps = %d", num_eval_steps)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            char_length=FLAGS.max_char_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_steps)

        result_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(result_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        if os.path.exists(os.path.join(FLAGS.output_dir, "pred.tf_record")) \
                and "num_pred_examples" in data_config.keys():
            pred_file = os.path.join(FLAGS.output_dir, "pred.tf_record")
            pred_examples = load_examples(predict_data_dir)
            num_pred_examples = data_config["num_pred_examples"]
            num_pred_steps = data_config["num_pred_steps"]
        else:
            pred_file = os.path.join(FLAGS.output_dir, "pred.tf_record")
            pred_examples = load_examples(predict_data_dir)
            num_pred_examples = filed_based_convert_examples_to_features(pred_examples, wptokenizer, pred_file)
            num_pred_steps = ceil(
                num_pred_examples / FLAGS.pred_batch_size)
            data_config["num_pred_examples"] = num_pred_examples
            data_config["num_pred_steps"] = num_pred_steps
            save_data_config(data_config)
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Num sentence examples = %d", num_pred_examples)
        tf.logging.info("  Batch size = %d", FLAGS.pred_batch_size)
        tf.logging.info("  Num steps = %d", num_pred_steps)


        predict_input_fn = file_based_input_fn_builder(
            input_file=pred_file,
            seq_length=FLAGS.max_seq_length,
            char_length=FLAGS.max_char_length,
            is_training=False,
            drop_remainder=False)
        prediction = estimator.predict(input_fn=predict_input_fn)

        id2tag_vocab = collections.OrderedDict()
        for k in tag_vocab.keys():
            id2tag_vocab[tag_vocab[k]] = k

        phi_dict = tokenization_ner.PHI_transform_Dict()

        for example in pred_examples:
            data = wptokenizer.tokenize(example.text)
            norm_data, orig_to_tok_index = data['norm_data'], data['orig_to_tok_index']
            pred_tags = []
            for orig_to_tok, pred in zip(orig_to_tok_index, prediction): # 此处要非常注意，较短的可迭代对象一定要在前面，不然会导致较长的缺一个
                pred_tags.append([id2tag_vocab[pred[idx]] for idx in orig_to_tok])
            items = wptokenizer.locate_items(norm_data['tokens'], pred_tags,
                                             norm_data['char_to_word_offset'])

            folder = os.path.join(FLAGS.output_dir, 'PHI-Gold-pred')
            if not os.path.exists(os.path.join(FLAGS.output_dir, 'PHI-Gold-pred')):
                os.makedirs(os.path.join(FLAGS.output_dir, 'PHI-Gold-pred'))

            file = os.path.join(folder, example.guid)
            root = etree.Element("deIdi2b2")

            textNode = etree.SubElement(root, 'TEXT')
            textNode.text = etree.CDATA(example.text)

            tagsNode = etree.SubElement(root, 'TAGS')

            for id, item in enumerate(items):
                attr = collections.OrderedDict()
                start, end = item['start'], item['end']
                attr['start'] = str(start)
                attr['end'] = str(end)
                attr['text'] = str(example.text[start: end])
                attr['TYPE'] = str(item['TYPE'])
                attr['id'] = 'P' + str(id)
                attr['comment'] = ""
                tNode = etree.SubElement(tagsNode, phi_dict[item['TYPE']], attrib=attr)

            tree = etree.ElementTree(root)
            tree.write(file, pretty_print=True, xml_declaration=True, encoding='utf-8')



def ner_evaluate(pred_examples, wptokenizer, id2tag_vocab, prediction):
    def get_orig_to_tok_and_tags(examples, tokenizer):  # 写一个生成器迭代生成orig_to_tok和tag_ids
        for example in examples:
            data = tokenizer.tokenize(example.text, example.tags)
            for s_id, orig_to_tok in enumerate(data["orig_to_tok_index"]):
                tag = data["tag_ids"][s_id] if data["tag_ids"] != None else None
                yield orig_to_tok, tag

    true_pred = get_orig_to_tok_and_tags(pred_examples, wptokenizer)

    pred, true = [], []
    for sing_pred, sing_true_pair in zip(prediction, true_pred):
        orig_to_tok = sing_true_pair[0]
        recover_pred = [id2tag_vocab[sing_pred[idx]] for idx in orig_to_tok]
        recover_true = [id2tag_vocab[sing_true_pair[1][idx]] for idx in orig_to_tok]
        pred.append(recover_pred)
        true.append(recover_true)

    evaluator = Evaluator(true, pred, tags=['PATIENT', 'DOCTOR', 'USERNAME',
                                            "PROFESSION",
                                            'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET',
                                            'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                                            'AGE',
                                            'DATE',
                                            'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
                                            'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
                                            'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM',
                                            'OTHER'])
    results, results_agg = evaluator.evaluate()

    res_str = json.dumps(results, indent=4)
    res_agg_str = json.dumps(results_agg, indent=4)
    output_predict_file = os.path.join(FLAGS.output_dir, "predict_results.json")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        writer.write(res_str)
        writer.write(res_agg_str)
    writer.close()

def save_data_config(data_config):
    with codecs.open(os.path.join(FLAGS.output_dir, "data_config.json"), 'w') as writer:
        data_config_file = json.dumps(data_config)
        writer.write(data_config_file)
    writer.close()


if __name__ == "__main__":
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()





