"""
Train / Test
Arguments:
    1. the word 'train' or 'test'
    2. words file-name
    3. data (restricted) file-name
    4. save file-name

Results (on test data): correct =  339151  total =  400000  percentage =  0.8478775

Author: Asher Stern
"""

import sys
import helper
import tensorflow as tf
from averager import Averager

word_filename = sys.argv[2]
data_filename = sys.argv[3]
print 'word_filename = ', word_filename
print 'data_filename = ', data_filename

batch_size = 350
lstm_size = 10
header_text_length = 10
content_text_length = 30
word_vector_size = 30
learning_rate = 0.01
decay = 0.5
decay_steps = 2000
number_of_batches = 10000

(word_to_index, index_to_word) = helper.load_word_map(word_filename)
number_of_words = len(word_to_index)

input_header_text = tf.placeholder(dtype=tf.int32, shape=[batch_size, header_text_length])
input_content_text = tf.placeholder(dtype=tf.int32, shape=[batch_size, content_text_length])
input_labels = tf.placeholder(dtype=tf.float32, shape=[batch_size])

word_vectors = tf.Variable(tf.concat([tf.zeros(shape=[1, word_vector_size], dtype=tf.float32), tf.random_normal(shape=[number_of_words, word_vector_size])], 0))
with tf.variable_scope("header"):
    header_cell = tf.contrib.rnn.LSTMCell(lstm_size)
with tf.variable_scope("content"):
    content_cell = tf.contrib.rnn.LSTMCell(lstm_size)
header_mul_weight = tf.Variable(tf.random_normal(shape=[1]))
content_mul_weight = tf.Variable(tf.random_normal(shape=[1]))

header_weight_vector = tf.Variable(tf.random_normal(shape=[lstm_size,1]))
content_weight_vector = tf.Variable(tf.random_normal(shape=[lstm_size,1]))
header_bias = tf.Variable(tf.zeros(shape=[1]))
content_bias = tf.Variable(tf.zeros(shape=[1]))

header_text_vectors = tf.nn.embedding_lookup(word_vectors, input_header_text) # batch_size * text_length * word_vector_size
content_text_vectors = tf.nn.embedding_lookup(word_vectors, input_content_text) # batch_size * text_length * word_vector_size
with tf.variable_scope("header"):
    (header_output, header_state) = tf.nn.dynamic_rnn(cell=header_cell, inputs=header_text_vectors, dtype=tf.float32)
    header_state_c = header_state.c  # batch_size * lstm_size
with tf.variable_scope("content"):
    (content_output, content_state) = tf.nn.dynamic_rnn(cell=content_cell, inputs=content_text_vectors, dtype=tf.float32)
    content_state_c = content_state.c  # batch_size * lstm_size

header_mul_result = tf.matmul(header_state_c, header_weight_vector) + header_bias
content_mul_result = tf.matmul(content_state_c, content_weight_vector) + content_bias
predicted = tf.sigmoid(header_mul_weight * header_mul_result + content_mul_weight * content_mul_result)
predicted_as_vector = tf.reshape(predicted, shape=[-1])
loss = tf.reduce_sum(tf.square(predicted - tf.reshape(input_labels, shape=[batch_size, 1])))

global_step = tf.Variable(0, trainable=False)
decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay, staircase=True)
train_op = tf.train.GradientDescentOptimizer(learning_rate=decayed_learning_rate).minimize(loss, global_step=global_step)

saver = tf.train.Saver()

init_op = tf.global_variables_initializer()


if __name__ == '__main__':
    train_or_test = sys.argv[1]
    save_file = sys.argv[4]
    print 'train_or_test = ', train_or_test
    print 'save_file = ', save_file
    if train_or_test == 'test':
        with tf.Session() as session:
            saver.restore(session, save_file)
            with helper.LineGroupProvider(data_filename, batch_size) as provider:
                correct = 0
                total = 0
                for p_lines in provider:
                    lines = helper.fill(p_lines, batch_size)
                    input_header_text_vectors = helper.convert_lines_to_matrix(lines, word_to_index, header_text_length, 'h')
                    input_content_text_vectors = helper.convert_lines_to_matrix(lines, word_to_index, content_text_length, 'c')
                    feed_dict = {input_header_text: input_header_text_vectors, input_content_text: input_content_text_vectors}
                    _predicted = session.run(predicted, feed_dict=feed_dict)
                    expected = helper.convert_lines_to_labels(lines)[:len(p_lines)]
                    (a_correct, _) = helper.prediction_assessment(expected, _predicted[:len(p_lines)])
                    correct += a_correct
                    total += len(p_lines)
                    print 'correct = ', correct, ' total = ', total, ' percentage = ', float(correct) / float(total)
                print 'final results:'
                print 'correct = ', correct, ' total = ', total, ' percentage = ', float(correct)/float(total)

    elif train_or_test == 'train':
        with open(data_filename) as data_file:
            with tf.Session() as session:
                session.run(init_op)
                av = Averager(50)
                for batch_index in range(number_of_batches):
                    lines = helper.read_file_in_loop(data_file, batch_size)
                    input_header_text_vectors = helper.convert_lines_to_matrix(lines, word_to_index, header_text_length, 'h')
                    input_content_text_vectors = helper.convert_lines_to_matrix(lines, word_to_index, content_text_length, 'c')
                    input_label_vector = helper.convert_lines_to_labels(lines)

                    feed_dict = {input_header_text: input_header_text_vectors, input_content_text: input_content_text_vectors, input_labels: input_label_vector}
                    (_loss, _, _predicted) = session.run([loss, train_op, predicted_as_vector], feed_dict=feed_dict)
                    assessment = helper.prediction_assessment(input_label_vector, _predicted)
                    (_, _percent) = assessment
                    av.add(_percent)
                    if (batch_index % 50) == 0 or batch_index == (number_of_batches-1):
                        print 'batch: ', batch_index, ' loss: ', _loss
                        print 'assessment: ', assessment
                        print 'Last 50 iterations average: ', av.average()
                        print ''
                saver.save(session, save_file)
    else:
        raise ValueError('Bad argument: ' + train_or_test)
