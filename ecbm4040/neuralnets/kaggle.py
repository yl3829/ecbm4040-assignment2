#!/usr/bin/env python
# ECBM E4040 Fall 2017 Assignment 2
# This script is intended for task 5 Kaggle competition. Use it however you want.

#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time
from ecbm4040.image_generator import ImageGenerator
####################################
# TODO: Build your own LeNet model #
####################################
class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, 1, 1, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

def my_LeNet(input_x, input_y,
              img_len=32, channel_num=3, output_size=10,
              conv_featmap=[6, 16], fc_units=[84],
              conv_kernel_size=[5, 5], pooling_size=[2, 2],
              l2_norm=0.01, seed=235):

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    layer = dict()
    in_channel = channel_num
    conv_out = input_x
    for index in range(len(conv_featmap)):
        out_channel = conv_featmap[index]
        layer['conv_layer_'+str(index)] = conv_layer(input_x = conv_out,
                                                          in_channel=in_channel,
                                                          out_channel=out_channel,
                                                          kernel_shape=conv_kernel_size[index],
                                                          rand_seed=seed,
                                                          index=index)
        
        layer['pooling_layer_'+str(index)] = max_pooling_layer(input_x= layer['conv_layer_'+str(index)] .output(),
                                                                    k_size=pooling_size[index],
                                                                    padding="VALID")
        in_channel = out_channel
        conv_out = layer['pooling_layer_'+str(index)].output()


    # flatten
    pool_shape =conv_out.get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(conv_out, shape=[-1, img_vector_length])

    # fc layer
    fc_out = flatten
    in_size = img_vector_length
    for index in range(len(fc_units)):
        layer['fc_layer_'+str(index)] = fc_layer(input_x= fc_out,
                                                 in_size=in_size,
                                                 out_size=fc_units[index],
                                                 rand_seed=seed,
                                                 activation_function=tf.nn.relu,
                                                 index=index)
        fc_out = layer['fc_layer_'+str(index)].output()
        in_size = fc_units[index]


    layer['fc_layer_'+str(len(fc_units))] = fc_layer(input_x=fc_out,
                                                     in_size=fc_units[-1],
                                                     out_size=output_size,
                                                     rand_seed=seed,
                                                     activation_function=None,
                                                     index=len(fc_units))

    # saving the parameters for l2_norm loss
    #conv_w = [conv_layer_0.weight]
    conv_w = [layer['conv_layer_'+str(index)].weight for index in range(len(conv_featmap))]
    #fc_w = [fc_layer_0.weight, fc_layer_1.weight]
    fc_w = [layer['fc_layer_'+str(index)].weight for index in range(len(fc_units)+1)]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.reduce_sum(tf.norm(w)) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        label = tf.one_hot(input_y, output_size)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=layer['fc_layer_'+str(len(fc_units))].output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return layer['fc_layer_'+str(len(fc_units))].output(), loss

####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################
def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num


def my_training(X_train, y_train, X_val, y_val, 
                conv_featmap=[6],
                fc_units=[84],
                conv_kernel_size=[5],
                pooling_size=[2],
                l2_norm=0.01,
                seed=235,
                learning_rate=1e-2,
                epoch=20,
                batch_size=245,
                verbose=False,
                pre_trained_model=None):
    
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    num_of_samples, height, width, channels = X_train.shape
    output_size = np.unique(y_train).shape[0]

    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, height, width, channels], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = my_LeNet(xs, ys,
                            img_len=32,
                            channel_num=channels,
                            output_size=output_size,
                            conv_featmap=conv_featmap,
                            fc_units=fc_units,
                            conv_kernel_size=conv_kernel_size,
                            pooling_size=pooling_size,
                            l2_norm=l2_norm,
                            seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1
                index = np.random.choice(X_train.shape[0],batch_size)

                training_batch_x = X_train[index]
                training_batch_y = y_train[index]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))

##########################################
#            End of your code            #
##########################################


def my_training_task4(X_train, y_train, X_val, y_val,
                      conv_featmap=[6],
                      fc_units=[84],
                      conv_kernel_size=[5],
                      pooling_size=[2],
                      l2_norm=0.01,
                      seed=235,
                      learning_rate=1e-2,
                      epoch=20,
                      batch_size=245,
                      verbose=False,
                      pre_trained_model=None):
    # TODO: Copy my_training function, make modifications so that it uses your
    # data generator from task 4 to train.
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))
    num_of_samples, height, width, channels = X_train.shape
    output_size = np.unique(y_train).shape[0]

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, height, width, channels], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = my_LeNet(xs, ys,
                           img_len=32,
                           channel_num=channels,
                           output_size=output_size,
                           conv_featmap=conv_featmap,
                           fc_units=fc_units,
                           conv_kernel_size=conv_kernel_size,
                           pooling_size=pooling_size,
                           l2_norm=l2_norm,
                           seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        
        #bat = batch_size/4
        print("Generating data")
        origin = ImageGenerator(X_train,y_train)
        #flip = ImageGenerator(X_train,y_train)
        #flip.flip(mode='h')
        #noise = ImageGenerator(X_train,y_train)
        #noise.add_noise(portion=1, amplitude=1)
        #rotate = ImageGenerator(X_train,y_train)
        #rotate.rotate(angle=45)
        translate = ImageGenerator(X_train,y_train)
        generator = []
        generator.append(origin.next_batch_gen(batch_size))
        #generator.append(flip.next_batch_gen(batch_size))
        #generator.append(noise.next_batch_gen(batch_size))
        #generator.append(rotate.next_batch_gen(batch_size))
        generator.append(translate.next_batch_gen(batch_size))
        print("Data generation finished")


        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))
            shift = np.random.choice(np.arange(1,11),size=2,replace=True)
            translate = ImageGenerator(X_train,y_train)
            translate.translate(shift_height=shift[0], shift_width=shift[1])
            generator[-1] = translate.next_batch_gen(batch_size)

            for itr in range(iters):
                iter_total += 1
                #index = np.random.choice(X_train.shape[0],batch_size)

                #training_batch_x = X_train[index]
                #training_batch_y = y_train[index]
                training_batch_x,training_batch_y = next(generator[itr%len(generator)])

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))


    
def prediction_on_test(X_test,model_name,batch_size=1000, mode='new'):
    with tf.Session() as sess: 
        try:
            saver = tf.train.Saver()
        except:
            saver = tf.train.import_meta_graph('model/'+model_name+'.meta')
        #saver.restore(sess, tf.train.latest_checkpoint('model/'))
        saver.restore(sess, 'model/'+model_name)
        graph = tf.get_default_graph()
                
        idx = 0
        tf_input = graph.get_operations()[idx].name+':0'
        x = graph.get_tensor_by_name(tf_input)
        
        pred = graph.get_tensor_by_name('evaluate/ArgMax:0')
        # Make prediciton
        num_of_samples = X_test.shape[0]
        num_of_batches = num_of_samples//batch_size+1
        out = []
        for bat in range(num_of_batches):
            y_out = sess.run(pred, feed_dict={x: X_test[bat*batch_size:min((bat+1)*batch_size,num_of_samples)]})
            out += list(y_out)
    return(out)
    