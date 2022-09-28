import argparse
import xml.etree.ElementTree as ET
import re
import json
import itertools
from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import numpy as np
import pickle

def args_parser():
  parser = argparse.ArgumentParser(description='Center Eigo solver of part 2')
  parser.add_argument('--question_file', type=str, default='data/center-2017-2015/dev/Center-2017--Main-Eigo_hikki.xml',
                      help='test file')
  parser.add_argument('--answer_file', type=str, default='data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki.json',
                      help='answer file')
  args = parser.parse_args()                    
  return args

def data_parser(node):
  s = ''.join(node.itertext()).strip()
  s = re.sub('[ 　]+', ' ', s)
  s = s.replace('\n', '')
  return s

def xml_parser(input_file):
  tree = ET.parse(input_file)
  root = tree.getroot()
  part_number = 0
  PART_NUMBER = 2
  dic_question = {}
  list_question = []

  for parts in root:
    # 対象とする大問番号を選択する
    if parts.tag == 'question':
      part_number += 1
      if part_number != PART_NUMBER:
        continue
    # print(parts.tag, parts.attrib)

    for middle_parts in parts:
      # print(middle_parts.tag, middle_parts.attrib)

      for question_element in middle_parts: 
        # print(question_element.tag, question_element.attrib)
        if question_element.tag == 'question' and question_element.attrib['minimal'] == 'yes': 
          dic_question = question_element.attrib
          # print(question_element.tag, question_element.attrib)

          for question_info in question_element:
            # print(question_info.tag,question_info.attrib)
            if (question_info.tag == 'data'):
              dic_question.setdefault('data', []).append(data_parser(question_info))
            elif (question_info.tag == 'choices'):
              for choice in question_info:
                dic_question.setdefault('choices', []).append(data_parser(choice))
              list_question.append(dic_question)
              dic_question = {}
            else :
              continue
  return list_question

def json_parser(input_file):
  with open(input_file) as f:
    list_answer = json.load(f)
  return list_answer

def data_parser(node):
  s = ''.join(node.itertext()).strip()
  s = re.sub('[ 　]+', ' ', s)
  s = s.replace('\n', '')
  return s

# それぞれの辞書にquestion_dataをキーとした入力用問題文を追加する
# それぞれの辞書にanswer_dataをキーとした正解文を追加する
def make_test_list(list_question, list_answer):
  list_test = []
  if len(list_question) != len(list_answer):
    print('list_question length is = ' + str(len(list_question)) + ' but,list_answer length is = ' + str(len(list_answer)))
    return -1
  for i, dic_question in enumerate(list_question):
    dic_question['question_data'] = convert_data_to_question(dic_question)
    dic_question['answer_data'] = convert_data_to_answer(dic_question, list_answer[i])
    list_test.append(dic_question)
  return list_test

def convert_data_to_question(dic_question):
  list_question_data = []
  if (dic_question['answer_type'] == 'sentence'):
    if (dic_question['anscol'].count('A') == 1):
      # 文法・語法・語彙
      data = dic_question['data'][0]
      anscol = dic_question['anscol']
      choices = dic_question['choices']

      anscol = anscol.replace('A', '')
      choices = [re.sub('[①-⑨] ', '', choice) for choice in choices]

      for choice in choices:
        question_data = data.replace(anscol, choice)
        question_data = re.sub('[ 　]+', ' ', question_data)
        list_question_data.append(question_data)

    elif (dic_question['anscol'].count('A') == 2):
      # 順序並び替え問題
      # dataに本文が入りきっていない
      data = dic_question['data'][0]
      choices = dic_question['choices']

      choices = [re.sub('[①-⑨] ', '', choice) for choice in choices]
      list_data = data.split(':')
      data = list_data[-1].strip()

      list_question = list(itertools.permutations(choices, len(choices)))
      for question_data in list_question:
        question_data = " ".join(question_data)
        question_data = re.sub('[ 　]+', ' ', question_data)
        question_data = re.sub('\s\d{,2}\s{3}\d{,2}\s', question_data, data)
        list_question_data.append(question_data)

  elif (dic_question['answer_type'] == '(symbol-sentence)*2'):
    # 文法・語法・語彙
    data = dic_question['data'][0]
    choices = dic_question['choices']
  
    for choice in choices:
      list_choice = choice.split(': ')
      choice_A = list_choice[1].replace(' B ', '')
      choice_B = list_choice[-1]
      question_data = data.replace('(A)', choice_A).replace('(B)', choice_B)
      question_data = re.sub('[ 　]+', ' ', question_data)
      list_question_data.append(question_data)

  elif (dic_question['answer_type'] == 'o(symbol-symbol-symbol)'):
    # 語句整除問題
    # dataに本文が入りきっていない
    replace_list_data = []
    list_first_data = []
    list_second_data = []
    list_third_data = []
    list_question = []
    list_question_data = []
    data_first = dic_question['data'][0]
    list_data_first = data_first.split(':')
    replace_data_first = list_data_first[-1].strip()
    data_second = dic_question['data'][1]
    anscol = dic_question['anscol']
    anscol = anscol.replace('A', '')
    choices = dic_question['choices']
    list_data = re.split('\(.\)', data_second)
    for replace_data in list_data:
      if replace_data == '':
        continue
      replace_data = replace_data.replace('→','').strip()
      replace_list_data.append(replace_data)
    list_first_data.append(replace_list_data[0])
    list_first_data.append(replace_list_data[3])
    list_second_data.append(replace_list_data[1])
    list_second_data.append(replace_list_data[4])
    list_third_data.append(replace_list_data[2])
    list_third_data.append(replace_list_data[5])

    list_question = list(itertools.product(list_first_data, list_second_data, list_third_data))
    for question_data in list_question:
        question_data = " ".join(question_data)
        question_data = re.sub('[ 　]+', ' ', question_data)
        question_data = replace_data_first.replace(anscol, question_data)
        list_question_data.append(question_data)
        print(question_data)
  else:
    print(str(dic_question['answer_type']) + ' is 未実装')
  return list_question_data

def convert_data_to_answer(dic_question, dic_answer):
  answer_data = ''
  list_answer = []
  if (dic_question['answer_type'] == 'sentence'):
    if (dic_question['anscol'].count('A') == 1):
      # 文法・語法・語彙
      answer_data = dic_question['question_data'][int(dic_answer['answer']) - 1]
    elif (dic_question['anscol'].count('A') == 2):
      # 順序並び替え問題
      list_answer = dic_answer['answer'].split('|')
      answer_data = dic_question['choices'][int(list_answer[0]) - 1] + '|' + dic_question['choices'][int(list_answer[1]) - 1]
      answer_data = re.sub('[①-⑨] ', '', answer_data)
  elif (dic_question['answer_type'] == '(symbol-sentence)*2'):
    # 文法・語法・語彙
    answer_data = dic_question['question_data'][int(dic_answer['answer']) - 1]
  elif (dic_question['answer_type'] == 'o(symbol-symbol-symbol)'):
    # 語句整除問題
    answer_data = dic_question['question_data'][int(dic_answer['answer']) - 1]
  else:
    print(str(dic_question['answer_type']) + ' is 未実装')
  return answer_data

def predict(list_test):
  ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]
  # MXNet MLMs (use names from mlm.models.SUPPORTED_MLMS)
  model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
  scorer = MLMScorer(model, vocab, tokenizer, ctxs)

  sentences = []
  list_result = []
  for dic in list_test:
    sentences = dic['question_data']
    result = np.array(scorer.score_sentences(sentences))
    list_result.append(result)
  file_name = './list_predict.txt'
  # pickle_save(list_result, file_name)
  return list_result

def calculate(list_test, list_result):
  file_name = './list_predict.txt'
  list_result = pickle_load(file_name)
  num = len(list_result)
  col = 0
  for i, dic_test in enumerate(list_test):
    result_index = list_result[i].argsort()[::-1]
    question_data = dic_test['question_data'][result_index[0]]
    if (dic_test['answer_type'] == 'sentence'):
      if (dic_test['anscol'].count('A') == 1):
        # 文法・語法・語彙
        if (dic_test['answer_data'] == question_data):
          col += 1
        else:
          print(dic_test['id'])
          print(dic_test['answer_data'])
          print(question_data)
      elif (dic_test['anscol'].count('A') == 2):
        # 順序並び替え問題
        list_answer = dic_test['answer_data'].split('|')
        list_question_data = question_data.split(' ')
        if (list_answer[0] == list_question_data[1] and list_answer[1] == list_question_data[4]):
          col += 1
        else:
          print(dic_test['id'])
          print(list_answer)
          print(list_question_data)
    elif (dic_test['answer_type'] == '(symbol-sentence)*2'):
      # 文法・語法・語彙
      if (dic_test['answer_data'] == question_data):
        col += 1
      else:
        print(dic_test['id'])
        print(dic_test['answer_data'])
        print(question_data)
    elif (dic_test['answer_type'] == 'o(symbol-symbol-symbol)'):
      # 語句整除問題
      if (dic_test['answer_data'] == question_data):
          col += 1
      else:
        print(dic_test['id'])
        print(dic_test['answer_data'])
        print(question_data)
    else:
      print(str(dic_test['answer_type']) + ' is 未実装')
  
  print('num = ' + str(num) + ' col = ' + str(col))
  return 0

def pickle_save(list, file_name):
  f = open(file_name, 'wb')
  pickle.dump(list_result, f)
  f.close()
  return 0

def pickle_load(file_name):
  f = open(file_name, 'rb')
  list_loaded = pickle.load(f)
  f.close()
  return list_loaded

if __name__ == "__main__":
  args = args_parser()
  list_question = []
  list_answer = []
  list_test = []
  list_result = []
  list_question = xml_parser(args.question_file)
  list_answer = json_parser(args.answer_file)
  list_test = make_test_list(list_question, list_answer)
  list_result = predict(list_test)
  calculate(list_test, list_result)