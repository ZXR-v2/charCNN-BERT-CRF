#！usr/bin/python
# -*- coding: utf-8 -*-
#==========================
import os
import collections
from lxml import etree
import xml.etree.ElementTree as ET
import unicodedata
from bert import tokenization
from nltk.tokenize import sent_tokenize

class xmlProcessor:
    def decodeXML(self, rootpath, xml):
        xmlpath = os.path.join(rootpath, xml)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        xml_text = ''
        xml_tags = []
        for child in root:
            if child.tag == 'TEXT':
                xml_text = child.text
                # print(child.text)
            elif child.tag == 'TAGS':
                for sub in child:
                    xml_tags.append(sub.attrib)
                    # print(sub.attrib)
        context = {}
        context['text'] = xml_text
        context['tags'] = xml_tags
        return context


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

# def extract_item()
#
#
# def replace(tokens, tokens_by_sent, tags):
#     rp_tokens = []
#     return rp_tokens


class NormTokenizer(object):

    def __init__(self, do_lower_case=False, cut_fine_grained=True):
        """Constructs a NormTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
          cut_fine_grained: Whether cut the word into small pieces accordinfg to punctuation
        """
        self.do_lower_case = do_lower_case
        self.cut_fine_grained = cut_fine_grained

    def tokenize(self, text, tags_list=None, test_id=None):
        """
        将文章切分为句子以及它们的token，此处的token使用的是空格和标点符号作为分割
        :return:
        """
        norm_tokens, char_to_word_offset = self._norm_tokenize(text)
        norm_sents = []
        all_sents = sent_tokenize(text)
        stpos = 0
        tok_num = 0
        for sent_id, sent in enumerate(all_sents):
            currpos = text[stpos:].find(sent)
            tokens, _ = self._norm_tokenize(sent)
            if not tokens == norm_tokens[tok_num: tok_num+len(tokens)]:
                # 把下一个句子的首字母添加到当前句子最后一个token的最后一位
                # 因为用sent_tokenize切分时，会出现sent_tokenize('???????!!!...You dare???')切分为['???????!!', '!...You dare??', '?']的情况
                # 因此这样就破坏了我们token的切分规则，因此将下一个句子的首位往当前句子最后一个token添加就行
                tokens[-1] += all_sents[sent_id+1][0]
                all_sents[sent_id+1] = all_sents[sent_id+1][1:]
            stpos = currpos + len(sent)
            norm_sents.append(tokens)
            tok_num += len(tokens)

        # 判断部分
        if len(norm_tokens) != tok_num:
            raise ValueError("The normal cut of the sentence is diffrent from "
                             "the text are cut into sentences and then normal cut"
                             " the text id is %s"%test_id)

        norm_sent_tags = None
        if tags_list != None:
            (norm_token_tags, norm_sent_tags) = self._get_norm_tags(tags_list, norm_tokens, char_to_word_offset, norm_sents)

        norm_data = {}
        norm_data['tokens'] = norm_tokens
        norm_data['tokens_by_sent'] = norm_sents
        norm_data['char_to_word_offset'] = char_to_word_offset
        norm_data['norm_tags'] = norm_sent_tags

        return norm_data

    def locate_item(self, tokens, tags, char_to_word_offset):
        """
        :param tokens:
        :param tokens_by_sent:
        :param tags:
        :param char_to_word_offset:
        :return: 按照xml的格式，返回敏感item的位置和种类
        """
        flat_tags = [y for x in tags for y in x]

        word_to_stchar_pos = []  # 装每个token的字符起始位置
        prev = -1 # 因为char_to_word_offset的word是从0开始的

        for pos, cur in enumerate(char_to_word_offset):
            if isinstance(cur, int):
                if cur != prev:
                    word_to_stchar_pos.append(pos)
                    prev = cur
            else:
                continue

        if len(flat_tags) != len(tokens):
            raise ValueError("the tags number %d is not the same as tokens %d"%(len(flat_tags), len(tokens)))
        assert len(word_to_stchar_pos) == len(tokens)

        items = []
        item = {}
        pre_tag = 'O'
        for idx, tag in enumerate(flat_tags):
            if tag == 'O':
                if pre_tag[0] != 'O':
                    # 上一token为实体结束，则上一个token的开始位置 + token长度为实体的结束位置
                    item['end'] = word_to_stchar_pos[idx - 1] + len(tokens[idx - 1])
                    items.append(item)
                pre_tag = 'O'
                continue
            elif tag[0] == 'B':
                if pre_tag != 'O':
                    item['end'] = word_to_stchar_pos[idx - 1] + len(tokens[idx - 1])  # 上一个token的开始位置 + token长度
                    items.append(item)
                item = {}
                item['TYPE'] = tag[2:]
                item['start'] = word_to_stchar_pos[idx]  # 当前token的start即为开始
                pre_tag = tag
            elif tag[0] == 'I':
                if pre_tag[2:] == tag[2:]:
                    continue
                else:
                    item = {}
                    item['TYPE'] = tag[2:]
                    item['start'] = word_to_stchar_pos[idx]  # 当前token的start即为开始
                    pre_tag = tag

        return items

    def _norm_tokenize(self, text):
        """
        按空格切分word与word之间，若self.cut_fine_grained=True，则会按字符的种类切分，比如标点符号与普通字符会区分开
        char_to_word_offset:每个字符对应的word的偏移量，例如：'  Hello\n\n\n\nI\'m Tracy.'
        对应的偏移量为[' ', ' ', 0, 0, 0, 0, 0, '\n', '\n', '\n', ''\n, 1, 2, 3, ' ', 4, 4, 4, 4, 4, 5]；
        norm_tokens:原始的未用分隔符切分的token，即word。如2069--04--07等。若切分则为2069，--，04，--，07；
        :return:
        """
        norm_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = False
        prev_is_punctuation = False
        prev_is_digit = False
        prev_is_alpha = False
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if is_whitespace(c):
                prev_is_whitespace = True
                prev_is_punctuation = False
                prev_is_digit = False
                prev_is_alpha = False
                char_to_word_offset.append(c) # 空格之类的直接用原空格占位
            else:
                if prev_is_whitespace:
                    norm_tokens.append(c)
                    prev_is_digit = True if str.isdigit(c) else False
                    prev_is_alpha = True if str.isalpha(c) else False
                    prev_is_punctuation = (True if is_punctuation(c) else False)
                elif is_punctuation(c) and self.cut_fine_grained:
                    if prev_is_punctuation and norm_tokens[-1][-1]==c: # punctuation要连续多个相同才视为一个，不同的就拆开来
                        norm_tokens[-1] += c
                    else:
                        norm_tokens.append(c)
                    prev_is_punctuation = True
                    prev_is_digit = False
                    prev_is_alpha = False
                else :
                    if prev_is_punctuation and self.cut_fine_grained:
                        norm_tokens.append(c)
                        prev_is_digit = True if str.isdigit(c) else False
                        prev_is_alpha = True if str.isalpha(c) else False
                    elif not prev_is_digit and str.isdigit(c):
                        norm_tokens.append(c)
                        prev_is_digit = True
                        prev_is_alpha = False
                    elif not prev_is_alpha and str.isalpha(c):
                        norm_tokens.append(c)
                        prev_is_digit = False
                        prev_is_alpha = True
                    else:
                        norm_tokens[-1] += c
                    prev_is_punctuation = False
                prev_is_whitespace = False
                char_to_word_offset.append(len(norm_tokens) - 1) # char所在token的序号作为该char的占位

        return (norm_tokens, char_to_word_offset)

    def _get_norm_tags(self, tags_list, norm_tokens, char_to_word_offset, norm_sents):
        """
        norm_tags:给norm_tokens打的标签
        norm_sent_tags同理
        :return:
        """
        norm_tags = ['O' for _ in range(len(norm_tokens))]
        for tag in tags_list:
            st = int(tag['start'])
            ed = int(tag['end']) - 1   #因为实体标注的end是token结束的后一个char，所以往前走一位char
            while str(char_to_word_offset[st]).strip()=='': # 因为数据集的实体中存在前后有空格这种不符合规格的情况，因此要找到第一个非空格的字符
                st += 1
            while str(char_to_word_offset[ed]).strip()=='':
                ed -= 1
            start_pos = char_to_word_offset[st]
            end_pos = char_to_word_offset[ed]
            assert end_pos != -1 #确保end_pos不等于-1
            norm_tags[start_pos] = 'B-' + tag['TYPE']
            for i in range(start_pos+1, end_pos+1):
                norm_tags[i] = 'I-' + tag['TYPE']

        norm_sent_tags = []
        st = 0
        for sent in norm_sents:
            norm_sent_tags.append(norm_tags[st:st+len(sent)])
            st += len(sent)

        return (norm_tags, norm_sent_tags)


class CharTokenizer(object):
    def __init__(self, max_char_len):
        self.max_char_len = max_char_len

    def tokenize(self, word='', pad="[PAD]"):
        """
        将完整的word传进去，获得切分后的字母以及字母列表
        :param word:
        :param max_len:
        :param pad:
        :return:
        """
        char_list = list(word)
        char_ids = []
        char_vocab = build_char_vocab()
        if len(char_list) > self.max_char_len:
            char_list = char_list[:self.max_char_len]
        else:
            for _ in range(len(char_list), self.max_char_len):
                char_list.append(pad)

        for char in char_list:
            char_ids.append(char_vocab.get(char, 0))

        return (char_list, char_ids)


class WPTokenizer(object):

    def __init__(self, vocab, max_seq_len, max_char_len, do_lower_case=False, cut_fine_grained=True):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.__norm_tokenizer = NormTokenizer(do_lower_case=do_lower_case, cut_fine_grained=cut_fine_grained)
        self.__wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text, tags_list=None, test_id=None):
        """
        使用wordpiece对norm token切分后得到的token
        :return:
        """
        norm_data = self.__norm_tokenizer.tokenize(text, tags_list, test_id)
        self.return_tag = norm_data['norm_tags'] != None
        new_tokens, new_sents, new_tags = [], [], None
        tok_to_orig_index, orig_to_tok_index = [], []

        input_ids, tag_ids= [], None
        input_mask, segment_ids = [], []
        type_vocab = None
        new_chars, char_ids = [], []
        if self.return_tag:
            new_tags, tag_ids = [], []
            type_vocab = build_tag_vocab()

        ch_tokenizer = CharTokenizer(self.max_char_len)

        norm_sents = norm_data['tokens_by_sent']
        norm_tags = norm_data['norm_tags']

        for sent_id, sent in enumerate(norm_sents):
            tok_to_orig, orig_to_tok= [], []
            new_sent, new_char, new_char_ids = [], [], []
            new_sent.append("[CLS]")
            new_char.append(ch_tokenizer.tokenize()[0])
            new_char_ids.append(ch_tokenizer.tokenize()[1])
            new_tag, eval_m = [], [] if self.return_tag else None
            if self.return_tag:
                new_tag.append("BOS")
            for word_id, word in enumerate(sent):
                sub_tokens = self.__wordpiece_tokenizer.tokenize(word)
                if len(new_sent)+len(sub_tokens) >= self.max_seq_len :
                    norm_sents.insert(sent_id+1, sent[word_id:])
                    if self.return_tag:
                        norm_tags.insert(sent_id+1, norm_tags[sent_id][word_id:])
                    break
                orig_to_tok.append(len(new_sent))
                (c, c_ids) = ch_tokenizer.tokenize(word)
                for idx, sub_token in enumerate(sub_tokens):
                    tok_to_orig.append(word_id)
                    new_sent.append(sub_token)
                    new_tokens.append(sub_token)
                    new_char.append(c)
                    new_char_ids.append(c_ids)
                    if self.return_tag:
                        if idx == 0:
                            new_tag.append(norm_tags[sent_id][word_id])
                        else:
                            new_tag.append('X')
            new_sent.append("[SEP]")
            new_char.append(ch_tokenizer.tokenize()[0]), new_char_ids.append(ch_tokenizer.tokenize()[1])
            if self.return_tag:
                new_tag.append("EOS")
            orig_to_tok_index.append(orig_to_tok)
            tok_to_orig_index.append(tok_to_orig)
            assert len(new_sent) == len(tok_to_orig) + 2
            mask = [1] * len(new_sent)

            if len(new_sent) > self.max_seq_len:
                raise ValueError("The sentence is longger than: %d" % (self.max_seq_len))
            while len(new_sent) < self.max_seq_len:
                new_sent.append("[PAD]")
                mask.append(0)
                new_char.append(ch_tokenizer.tokenize()[0]), new_char_ids.append(ch_tokenizer.tokenize()[1])
                if self.return_tag:
                    new_tag.append("O")

            assert len(new_sent) == self.max_seq_len
            assert len(new_char) == self.max_seq_len
            assert len(new_char_ids) == self.max_seq_len

            new_sents.append(new_sent), new_chars.append(new_char)
            char_ids.append(new_char_ids)
            input_ids.append(tokenization.convert_tokens_to_ids(self.vocab, new_sent))
            input_mask.append(mask)
            segment_ids.append([0]*self.max_seq_len)
            if self.return_tag:
                assert len(new_sent) == len(new_tag)
                new_tags.append(new_tag)
                tag_ids.append(tokenization.convert_tokens_to_ids(type_vocab, new_tag))

        assert len(new_sents) == len(tok_to_orig_index)
        assert len(new_sents) == len(orig_to_tok_index)
        if self.return_tag:
            assert len(new_sents) == len(new_tags)

        wp_data = {}
        wp_data['tokens'] = new_tokens
        wp_data['tokens_by_sent'] = new_sents
        wp_data['tok_to_orig_index'] = tok_to_orig_index
        wp_data['orig_to_tok_index'] = orig_to_tok_index
        wp_data['norm_data'] = norm_data # 包括了切分前的tokens、tags和char_to_word_offset
        wp_data['wp_tags'] = new_tags
        wp_data["input_ids"] = input_ids
        wp_data["input_mask"] = input_mask
        wp_data["segment_ids"] = segment_ids
        wp_data["tag_ids"] = tag_ids
        wp_data["chars"] = new_chars
        wp_data["char_ids"] = char_ids

        return wp_data

    def recover_tags(self, orig_to_tok_index, wp_tags):
        """
        WordPiece切分后的token所对应的tag给拼接回原来的norm形式的模样，因为wordpiece对应的tag有X这个无关tag
        :param tokens:
        :param tokens_by_sent: 原本想把tokens_by_sent中被切开的token重新拼接，其实在一开始保存一下就行
        :param tok_to_orig_index:
        :param orig_to_tok_index:
        :param wp_tags:
        :return:
        """
        # sent_comb, tag_comb = [], []
        # for sent, tok_to_orig, orig_to_tok, tag in zip(tokens_by_sent, tok_to_orig_index, orig_to_tok_index, wp_tags):
        #     assert len(tok_to_orig) == len(tag)
        #     for st_pos in orig_to_tok:
        #         tid = tok_to_orig[st_pos]
        #         new_tok, new_tag = sent[st_pos], tag[st_pos]
        #         achr = st_pos + 1
        #         while tid == tok_to_orig[achr]:
        #             new_tok += sent[achr][2:]
        #             achr += 1
        #         tag_comb.append(new_tag)
        #
        # return tag_comb
        tag_comb = []
        for orig_to_tok, tag in zip(orig_to_tok_index, wp_tags):
            tt = []
            for st_pos in orig_to_tok:
                tt.append(tag[st_pos])
            tag_comb.append(tt)

        return tag_comb

    def locate_items(self, tokens, tags, char_to_word_offset):
        return self.__norm_tokenizer.locate_item(tokens, tags, char_to_word_offset)

def build_wordpiece_vocab(root_path, bert_path, vocab_file):
    vocab_file = os.path.join(root_path, bert_path, vocab_file)
    vocab = tokenization.load_vocab(vocab_file)
    return vocab


def build_tag_vocab():
    type_list = ['PATIENT', 'DOCTOR', 'USERNAME',
                 "PROFESSION",
                 'ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET',
                  'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER',
                 'AGE',
                 'DATE',
                 'PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR',
                 'SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
                  'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM',
                 'OTHER']  # 从i2b2_evaluation_scripts的tags文件中截取的

    type_vocab = collections.OrderedDict()
    type_vocab['O'] = 0
    for i, type in enumerate(type_list):
        type_vocab['B-'+type] = 2 * i + 1
        type_vocab['I-'+type] = 2 * i + 2
    type_vocab['BOS'] = len(type_vocab)
    type_vocab['EOS'] = len(type_vocab)
    type_vocab['X'] = len(type_vocab)

    return type_vocab

def build_char_vocab():
    char_list = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_vocab = collections.OrderedDict()
    char_vocab['[UNK]'] = 0
    char_vocab['[PAD]'] = 1
    for char in char_list:
        char_vocab[char] = len(char_vocab)
        if char_vocab.get(char.upper(), -1) == -1:
            char_vocab[char.upper()] = len(char_vocab)

    return char_vocab


def PHI_transform_Dict():
    """
    transform i2b2 PHI to HIPAA PHI
    :return:
    """
    phi_dict = collections.OrderedDict()
    for phi in ['PATIENT', 'DOCTOR', 'USERNAME']:
        phi_dict[phi] = 'NAME'
    for phi in ["PROFESSION"]:
        phi_dict[phi] = 'PROFESSION'
    for phi in ['ROOM', 'DEPARTMENT', 'HOSPITAL', 'ORGANIZATION', 'STREET',
                  'CITY', 'STATE', 'COUNTRY', 'ZIP', 'LOCATION-OTHER']:
        phi_dict[phi] = 'LOCATION'
    for phi in ['AGE']:
        phi_dict[phi] = 'AGE'
    for phi in ['DATE']:
        phi_dict[phi] = 'DATE'
    for phi in ['PHONE', 'FAX', 'EMAIL', 'URL', 'IPADDR']:
        phi_dict[phi] = 'CONTACT'
    for phi in ['SSN', 'MEDICALRECORD', 'HEALTHPLAN', 'ACCOUNT',
                  'LICENSE', 'VEHICLE', 'DEVICE', 'BIOID', 'IDNUM']:
        phi_dict[phi] = 'ID'
    for phi in ['OTHER']:
        phi_dict[phi] = 'OTHER'

    return phi_dict



if __name__=="__main__":
    root = 'E:\\毕设项目\\deid_Bert-Plus_en'
    proc = xmlProcessor()
    _xml = proc.decodeXML(root, os.path.join(root, '110-01.xml'))
    text = '  hello\n\n\n\nI\'m 2072-07-22'
    print(sent_tokenize('???????!!!...You dare???'))
    # # print(sent_tokenize(xml['text']))
    normtokenizer = NormTokenizer()
    print(normtokenizer.tokenize(text))
    chartokenizer = CharTokenizer(16)
    print(chartokenizer.tokenize("AaI /seeS/S\\p\\F"))

    bert_path = 'multi_cased_L-12_H-768_A-12'
    vocab_file = os.path.join(root, bert_path, 'vocab.txt')
    vocab = tokenization.load_vocab(vocab_file)
    wptokenizer = WPTokenizer(vocab, max_seq_len=128, max_char_len=16)
    wp_data = wptokenizer.tokenize(_xml['text'], _xml['tags'])
    # print(wp_data)
    wp_tags = wptokenizer.recover_tags(wp_data['orig_to_tok_index'], wp_data['wp_tags'])

    # print(wptokenizer.char_tokenize(norm_data['tokens_by_sent'], wp_data['tok_to_orig_index']))
    print(wptokenizer.locate_items(wp_data['norm_data']['tokens'], wp_tags, wp_data['norm_data']['char_to_word_offset']))




