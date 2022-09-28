import argparse
import xml.etree.ElementTree as ET
import re
import json
import itertools
# from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
# from mlm.models import get_pretrained
# import mxnet as mx
import numpy as np
import pickle
import os

class InputExample:
    def __init__(self, paragraph, qa_list, label):
        self.paragraph = paragraph
        self.qa_list = qa_list
        self.label = label

def args_parser():
  parser = argparse.ArgumentParser(description='Center Eigo solver of part 5')
  parser.add_argument('-q', '--question_file', type=str, default='data/center-2017-2015/dev/Center-2017--Main-Eigo_hikki.xml',
                      help='test file')
  parser.add_argument('-a', '--answer_file', type=str, default='data/center-2017-2015/answer/Center-2017--Main-Eigo_hikki_part5.json',
                      help='answer file')
  parser.add_argument('-o', "--output_dir", type=str, default='output',
                      help="output directory for extracted data")
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
  PART_NUMBER = 5
  dic_question = {}
  dic_body = {}
  list_question = []

  for parts in root:
    # 対象とする大問番号を選択する
    if parts.tag == 'question':
      part_number += 1
      if part_number != PART_NUMBER:
        continue
    for middle_parts in parts:
      # print(middle_parts.tag,middle_parts.attrib)
      dic_question = middle_parts.attrib
      for question_element in middle_parts: 
        # print(question_element.tag,question_element.attrib)
        for question_info in question_element.iter('paragraph'):
          dic_body.setdefault('body', []).append(data_parser(question_info))
        if question_element.tag == 'data':
          dic_question.setdefault('data', []).append(data_parser(question_element))
          # print(data_parser(question_element)) 
        elif question_element.tag == 'choices':
          for choice in question_element:
            dic_question.setdefault('choices', []).append(data_parser(choice))
          dic_question.update(dic_body)
          list_question.append(dic_question)
          dic_question = {}
        else :
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
  for i in range(len(list_answer)):
    list_question[i].update(list_answer[i])
  return list_question

def logger(list_test, pickle=False):
  file_name = './list_predict.txt'
  if (pickle==True):
    list_result = pickle_load(file_name)
  for i, dic_test in enumerate(list_test):
    result_index = list_result[i].argsort()[::-1]
    question_data = dic_test['question_data'][result_index[0]]
    if (dic_test['answer_type'] == 'sentence'):
      if (dic_test['anscol'].count('A') == 1):
        # 文法・語法・語彙
        print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
      elif (dic_test['anscol'].count('A') == 2):
        # 順序並び替え問題
        print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    elif (dic_test['answer_type'] == '(symbol-sentence)*2'):
      # 文法・語法・語彙
      print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    elif (dic_test['answer_type'] == 'o(symbol-symbol-symbol)'):
      # 語句整除問題
      print(dic_test['id'] + '\t' + question_data + '\t' + dic_test['answer_data'])
    else:
      print(str(dic_test['answer_type']) + ' is Unimplemented')
  return 0

def make_dataset(input_list):
  examples = []
  # 制御文字\が入っている
  for dic in input_list:
    article = ''.join(dic['body'])
    question = ''.join(dic['data'])
    options = dic['choices']
    answer = dic['answer']
    anscol = dic['anscol']
    anscol = anscol.replace('A', '')
    qa_cat = ''
    qa_list = []
    for i in range(len(options)):
      option = options[i]
      option = re.sub('[①-⑨] ', '', option)
      if re.search(anscol,question):
        qa_cat = re.sub(anscol, option, question)
      else:
        qa_cat = ' '.join([question, option])
      qa_list.append(qa_cat)
    examples.append(InputExample(article, qa_list, answer))
  return examples

def write_dataset(input_list):
  examples = input_list
  qa_file_paths = [
      os.path.join(args.output_dir, "center.input" + str(i + 1))
      for i in range(4)
  ]
  qa_files = [open(qa_file_path, "w", encoding='utf-8') for qa_file_path in qa_file_paths]
  outf_context_path = os.path.join(args.output_dir, "center.input0")
  outf_label_path = os.path.join(args.output_dir, "center.label")
  outf_context = open(outf_context_path, "w", encoding='utf-8')
  outf_label = open(outf_label_path, "w", encoding='utf-8')
  for example in examples:
      outf_context.write(example.paragraph + "\n")
      for i in range(4):
          qa_files[i].write(example.qa_list[i] + "\n")
      outf_label.write(str(example.label) + "\n")
  for f in qa_files:
      f.close()
  outf_label.close()
  outf_context.close()
  return examples

if __name__ == "__main__":
  args = args_parser()
  list_question = []
  list_answer = []
  list_test = []
  dataset = []
  list_question = xml_parser(args.question_file)
  list_answer = json_parser(args.answer_file)
  list_test = make_test_list(list_question, list_answer)
  os.makedirs(args.output_dir, exist_ok=True)
  dataset = make_dataset(list_test)
  write_dataset(dataset)