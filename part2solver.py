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
  parser.add_argument('-q', '--question_file', type=str, default='data/center-2017-2015/dev/Center-2017--Main-Eigo_hikki.xml',
                      help='test file')
  parser.add_argument('-a', '--answer_file', type=str, default='data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki_part2.json',
                      help='answer file')
  args = parser.parse_args()                    
  return args

def data_parser(node):
  data = ''.join(node.itertext()).strip()
  data = re.sub('[ 　]+', ' ', data)
  data = data.replace('\n', '')
  return data

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
    for middle_parts in parts:
      for question_element in middle_parts: 
        if question_element.tag == 'question': 
          dic_question = question_element.attrib
          for question_info in question_element:
            if (question_info.tag == 'data'):
              dic_question.setdefault('data', []).append(data_parser(question_info))
            elif (question_info.tag == 'choices'):
              for choice in question_info:
                dic_question.setdefault('choices', []).append(data_parser(choice))
              list_question.append(dic_question)
              dic_question = {}
            else :
              continue
        else:
          continue
  return list_question

def json_parser(input_file):
  with open(input_file) as f:
    list_answer = json.load(f)
  return list_answer

def make_test_list(list_question, list_answer):
  if len(list_question) != len(list_answer):
    print('list_question length is = ' + str(len(list_question)) + ' but,list_answer length is = ' + str(len(list_answer)))
    return -1
  for i in range(len(list_question)):
    list_question[i]['question_data'] = convert_data_to_question(list_question[i])
    list_question[i]['answer_data'] = convert_data_to_answer(list_question[i], list_answer[i])
  return list_question
  
def convert_question_part2A_one_answer(dic_question):
  list_question_data = []
  data = dic_question['data'][0]
  anscol = dic_question['anscol']
  choices = dic_question['choices']

  anscol = anscol.replace('A', '')
  choices = [re.sub('[①-⑨] ', '', choice) for choice in choices]

  for choice in choices:
    question_data = data.replace(anscol, choice)
    question_data = re.sub('[ 　]+', ' ', question_data)
    list_question_data.append(question_data)
  return list_question_data

def convert_question_part2A_two_answers(dic_question):
  list_question_data = []
  data = dic_question['data'][0]
  choices = dic_question['choices']

  for choice in choices:
    list_choice = choice.split(': ')
    choice_A = list_choice[1].replace(' B ', '')
    choice_B = list_choice[-1]
    question_data = data.replace('(A)', choice_A).replace('(B)', choice_B)
    question_data = re.sub('[ 　]+', ' ', question_data)
    list_question_data.append(question_data)
  return list_question_data

def convert_question_part2B(dic_question):
  list_question_data = []
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
  return list_question_data

def convert_question_part2C(dic_question):
  list_question_data = []
  replace_list_choice = []
  list_first_data = []
  list_second_data = []
  list_third_data = []
  list_question = []
  list_question_data = []
  data = dic_question['data'][0]
  choice = dic_question['data'][1]
  anscol = dic_question['anscol']

  list_data = data.split(':')
  data = list_data[-1].strip()
  anscol = anscol.replace('A', '')
  list_choice = re.split('\(.\)', choice)
  for i in range(len(list_choice)):
    if list_choice[i] == '':
      continue
    replace_choice = list_choice[i].replace('→','').strip()
    replace_list_choice.append(replace_choice)
  list_first_data.append(replace_list_choice[0])
  list_first_data.append(replace_list_choice[3])
  list_second_data.append(replace_list_choice[1])
  list_second_data.append(replace_list_choice[4])
  list_third_data.append(replace_list_choice[2])
  list_third_data.append(replace_list_choice[5])

  list_question = list(itertools.product(list_first_data, list_second_data, list_third_data))
  for question_data in list_question:
      question_data = " ".join(question_data)
      question_data = re.sub('[ 　]+', ' ', question_data)
      question_data = data.replace(anscol, question_data)
      list_question_data.append(question_data)
  return list_question_data

def convert_data_to_question(dic_question):
  list_question_data = []
  if (dic_question['answer_type'] == 'sentence'):
    if (dic_question['anscol'].count('A') == 1):
      # 文法・語法・語彙
      list_question_data = convert_question_part2A_one_answer(dic_question)
    elif (dic_question['anscol'].count('A') == 2):
      # 順序並び替え問題
      list_question_data = convert_question_part2B(dic_question)
  elif (dic_question['answer_type'] == '(symbol-sentence)*2'):
    # 文法・語法・語彙
    list_question_data = convert_question_part2A_two_answers(dic_question)
  elif (dic_question['answer_type'] == 'o(symbol-symbol-symbol)'):
    # 語句整除問題
    list_question_data = convert_question_part2C(dic_question)
  else:
    print(str(dic_question['answer_type']) + ' is Unimplemented')
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
    print(str(dic_question['answer_type']) + ' is Unimplemented')
  return answer_data

def predict(list_test):
  ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]
  model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
  scorer = MLMScorer(model, vocab, tokenizer, ctxs)
  sentences = []
  list_result = []
  for dic in list_test:
    sentences = dic['question_data']
    result = np.array(scorer.score_sentences(sentences))
    list_result.append(result)
  return list_result

def calculate(list_test, list_result):
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
          print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
      elif (dic_test['anscol'].count('A') == 2):
        # 順序並び替え問題
        list_answer_number = dic_test['answer_data'].split('|')
        data = dic_test['data'][0]
        choices = dic_test['choices']
        list_answer_data = []

        choices = [re.sub('[①-⑨] ', '', choice) for choice in choices]
        list_data = data.split(':')
        data = list_data[-1].strip()

        list_answer = list(itertools.permutations(choices, len(choices)))
        for answer_data in list_answer:
          if (answer_data[1] == list_answer_number[0] and answer_data[4] == list_answer_number[1]):
            answer_data = " ".join(answer_data)
            answer_data = re.sub('[ 　]+', ' ', answer_data)
            answer_data = re.sub('\s\d{,2}\s{3}\d{,2}\s', answer_data, data)
            list_answer_data.append(answer_data)
        if (question_data in list_answer_data):
          col += 1
        else:
          print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    elif (dic_test['answer_type'] == '(symbol-sentence)*2'):
      # 文法・語法・語彙
      if (dic_test['answer_data'] == question_data):
        col += 1
      else:
        print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    elif (dic_test['answer_type'] == 'o(symbol-symbol-symbol)'):
      # 語句整除問題
      if (dic_test['answer_data'] == question_data):
          col += 1
      else:
        print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    else:
      print(str(dic_test['answer_type']) + ' is Unimplemented')
  print('num = ' + str(num) + ' col = ' + str(col))
  return 0

if __name__ == "__main__":
  args = args_parser()
  # 問題文
  list_question = []
  # 解答
  list_answer = []
  # 推論用データリスト
  list_test = []
  # 推論結果
  list_result = []

  list_question = xml_parser(args.question_file)
  list_answer = json_parser(args.answer_file)
  list_test = make_test_list(list_question, list_answer)
  list_result = predict(list_test)
  calculate(list_test, list_result)