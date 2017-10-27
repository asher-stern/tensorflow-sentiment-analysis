"""
Preprocess train and test files.
Dataset - Amazon review dataset, available in Kaggle.

Command line arguments:
    1. data file path
    2. target directory
    3. For test data only: path of words.txt file, generated when preprocessing the training data.

Author: Asher Stern
"""


import sys
import os
import string
from collections import Counter


def split_line(line):
    space_index = string.find(line, ' ')
    if space_index < 0:
        label = line.strip()
        header = ''
        content = ''
    else:
        label = line[:space_index]
        rest = line[space_index + 1:]
        if len(rest.strip()) == 0:
            header = ''
            content = ''
        else:
            dot_index = string.find(rest, ':')
            if dot_index < 0:
                raise ValueError('Bad line: '+line)
            header = rest[:dot_index].strip()
            content = rest[dot_index + 1:].strip()
    return (label, header, content)


def delete_non_letters(s):
    return ''.join([l if l == ' ' or (l in string.letters) else '' for l in s]).strip()


def normalize_line(l):
    (label, header, content) = split_line(l)
    return label+' '+delete_non_letters(string.lower(header))+': '+delete_non_letters(string.lower(content))


def normalize_file(source, target):
    with open(target, 'w') as t:
        with open(source) as s:
            for line in s:
                t.write(normalize_line(line) + '\n')


def text_of_line(l):
    (_, header, contents) = split_line(l)
    return header + ' ' + contents


def count_line_words(counter, l):
    counter.update(text_of_line(l).split())


def count_file_words(data_filename, words_filename):
    with open(data_filename) as data_file:
        with open(words_filename, 'w') as words_file:
            counter = Counter()
            for line in data_file:
                count_line_words(counter, line)
            sorted_words = [w for (w, c) in counter.most_common(10000)]
            for word in sorted_words:
                words_file.write(word+'\n')


def load_file_to_set(filename):
    with open(filename) as f:
        return {word for word in f.read().splitlines()}


def restrict_text_to_word_set(word_set, text):
    list_words = []
    for word in text.split():
        if word in word_set:
            list_words.append(word)
    return ' '.join(list_words)


def restrict_line_to_word_set(word_set, line):
    (label, header, contents) = split_line(line)
    return label + ' ' + restrict_text_to_word_set(word_set, header)+': '+restrict_text_to_word_set(word_set, contents)


def restrict_file_to_word_set(word_set, source, target):
    with open(source) as s_f:
        with open (target, 'w') as t_f:
            for line in s_f:
                t_f.write(restrict_line_to_word_set(word_set, line)+'\n')


def preprocess(filepath, target_directory, known_word_filepath = None):
    filename = os.path.basename(filepath)

    dot_index = string.rfind(filename, '.')
    filename_noext = filename[:dot_index]
    extension = filename[dot_index+1:]

    filename_normalized = os.path.join(target_directory, filename_noext + '_normalized' + '.' + extension)
    print 'Generate a normalized file: ', filename_normalized
    normalize_file(filepath, filename_normalized)

    if known_word_filepath is None:
        filename_words = os.path.join(target_directory, 'words' + '.' + extension)
        print 'Generate words file: ', filename_words
        count_file_words(filename_normalized, filename_words)
        word_set = load_file_to_set(filename_words)
    else:
        print 'Loading words file: ', known_word_filepath
        word_set = load_file_to_set(known_word_filepath)

    filename_restricted = os.path.join(target_directory, filename_noext + '_restricted' + '.' + extension)
    print 'Generate a restricted file: ', filename_restricted
    restrict_file_to_word_set(word_set, filename_normalized, filename_restricted)


if __name__ == '__main__':
    filepath = sys.argv[1]
    target_directory = sys.argv[2]
    if len(sys.argv) > 3:
        known_word_filepath = sys.argv[3]
        preprocess(filepath, target_directory, known_word_filepath)
    else:
        preprocess(filepath, target_directory)
