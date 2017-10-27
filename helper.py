import string
import preprocess


def read_file_in_loop(f, number_of_lines):
    ret = list()
    for _ in range(number_of_lines):
        line = f.readline().strip()
        if line == '':
            f.seek(0)
            line = f.readline().strip()
        ret.append(line)
    return ret


class LineGroupProvider:
    def __init__(self, filename, number_of_lines):
        self.file = open(filename)
        self.number_of_lines = number_of_lines
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        return self.provide()

    def provide(self):
        while not self.finished:
            lines = []
            for i in range(self.number_of_lines):
                line = self.file.readline().strip()
                if line == '':
                    self.finished = True
                    break
                lines.append(line)
            if len(lines) > 0:
                yield lines



def load_word_map(filename):
    with open(filename) as f:
        l = f.read().splitlines()
        word_to_index = dict()
        index_to_word = dict()
        word_to_index['__UNKNOWN__'] = 0
        index_to_word[0] = '__UNKNOWN__'
        index = 1
        for line in l:
            word = line.strip()
            if word == '':
                break
            word_to_index[word] = index
            index_to_word[index] = word
            index += 1
        return (word_to_index, index_to_word)


def convert_lines_to_matrix(lines, word_to_index, text_length, h_or_c):
    matrix = []
    for line in lines:
        (_, header, contents) = preprocess.split_line(line)
        if h_or_c == 'h':
            text = header
        elif h_or_c == 'c':
            text = contents
        else:
            raise ValueError('Bad argument h_or_c: '+h_or_c)
        words = text.split()
        vector = []
        for i in range(text_length):
            if i < len(words):
                vector.append(word_to_index[words[i]])
            else:
                vector.append(0)
        matrix.append(vector)
    return matrix


def convert_lines_to_labels(lines):
    labels = []
    for line in lines:
        (label, _, _) = preprocess.split_line(line)
        if label == '__label__1':
            labels.append(0)
        elif label == '__label__2':
            labels.append(1)
        else:
            raise ValueError('Bad label: '+label)
    return labels


def prediction_assessment(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError("len(expected) != len(predicted)")
    correct = 0
    for i in range(len(expected)):
        p = 0
        if predicted[i] > 0.5:
            p = 1
        if p == expected[i]:
            correct += 1
    return (correct, float(correct)/float(len(expected)))


def predictions_to_01(predictions):
    return [1 if p > 0.5 else 0 for p in predictions]


def fill(l, size):
    if len(l) == 0:
        raise ValueError("Empty list")
    if len(l) >= size:
        return l

    ll = len(l)
    ret = []
    ret.extend(l)
    for _ in range(size-ll):
        ret.append(l[0])
    return ret

