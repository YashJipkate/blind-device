import tensorflow as tf
import numpy as np
import os
import tqdm
import pandas as pd
from tqdm import tqdm
import string
from nltk.tokenize import word_tokenize

class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture\\
        self.image_file=r'C:\Users\ANSHUL\Desktop\files\photo\256_ObjectCategories\001.ak47\001_0008.jpg'
        self.model_file='./289999.npy'
        self.ilsvrc='./ilsvrc_2012_mean.npy'
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        self.max_caption_length = 20
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024
        self.beam_size=3

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 32
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'
        self.vocabulary_size = 5000

        # about the training
##        self.train_image_dir = './train/images/'
##        self.train_caption_file = './train/captions_train2014.json'
##        self.temp_annotation_file = './train/anns.csv'
##        self.temp_data_file = './train/data.npy'
##
##        # about the evaluation
##        self.eval_image_dir = './val/images/'
##        self.eval_caption_file = './val/captions_val2014.json'
##        self.eval_result_dir = './val/results/'
##        self.eval_result_file = './val/results.json'
##        self.save_eval_result_as_image = False
##
##        # about the testing
##        self.test_image_dir = './test/images/'
##        self.test_result_dir = './test/results/'
##        self.test_result_file = './test/results.csv'



import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as layers

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = False
        self.train_cnn = False
        self.prepare()

    def prepare(self):
        """ Setup the weight initalizers and regularizers. """
        config = self.config

        self.conv_kernel_initializer = layers.xavier_initializer()

        if self.train_cnn and config.conv_kernel_regularizer_scale > 0:
            self.conv_kernel_regularizer = layers.l2_regularizer(
                scale = config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if self.train_cnn and config.conv_activity_regularizer_scale > 0:
            self.conv_activity_regularizer = layers.l1_regularizer(
                scale = config.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):
        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.train_cnn,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):
        """ 2D Max Pooling layer. """
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        """ Dropout layer. """
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)

    def batch_norm(self,
                   inputs,
                   name = None):
        """ Batch normalization layer. """
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )





class BaseModel(object):
    def __init__(self, config):
        self.config = config
        #self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader(config.ilsvrc)
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0,
                                       name = 'global_step',
                                       trainable = False)
        self.build()

    def build(self):
        raise NotImplementedError()


    
    def test(self, sess, test_data, vocabulary):
        """ Test the model using any given images. """
        print("Testing the model ...")
        config = self.config

        #if not os.path.exists(config.test_result_dir):
        #    os.mkdir(config.test_result_dir)

        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(test_data.num_batches)), desc='path'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if k<test_data.num_batches-1 \
                         else test_data.fake_count
            for l in range(test_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                print(caption)
                #image_file = batch[l]
                #image_name = image_file.split(os.sep)[-1]
                #image_name = os.path.splitext(image_name)[0]
                #img = plt.imread(image_file)
                #plt.imshow(img)
                #plt.axis('off')
                #plt.title(caption)
                #plt.savefig(os.path.join(config.test_result_dir,
                 #                        'image_name'+'_result.jpg'))

        # Save the captions to a file
        #results = pd.DataFrame({'image_files':test_data.image_files,
                  #              'caption':captions,
                  #              'prob':scores})
        #results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, image_files, vocabulary):
        """Use beam search to generate the captions for a batch of images."""
        # Feed in the images to get the contexts and the initial LSTM states
        config = self.config
        images = self.image_loader.load_images(image_files)
        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict = {self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence = [],
                                       memory = initial_memory[k],
                                       output = initial_output[k],
                                       score = 1.0)
            partial_caption_data.append(TopN(3))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(3))

        # Run beam search
        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1]
                                        for pcl in partial_caption_data_lists],
                                        np.int32)

                last_memory = np.array([pcl[b].memory
                                        for pcl in partial_caption_data_lists],
                                        np.float32)
                last_output = np.array([pcl[b].output
                                        for pcl in partial_caption_data_lists],
                                        np.float32)

                memory, output, scores = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict = {self.contexts: contexts,
                                 self.last_word: last_word,
                                 self.last_memory: last_memory,
                                 self.last_output: last_output})

                # Find the beam_size most probable next words
                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[0:4+1]

                    # Append each of these words to the current partial caption
                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence,
                                           memory[k],
                                           output[k],
                                           score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results

    def save(self):
        """ Save the model. """
        config = self.config
        data = {v.name: v.eval() for v in tf.global_variables()}
        save_path = os.path.join(config.save_dir, str(self.global_step.eval()))

        print((" Saving the model to %s..." % (save_path+".npy")))
        np.save(save_path, data)
        info_file = open(os.path.join(config.save_dir, "config.pickle"), "wb")
        config_ = copy.copy(config)
        config_.global_step = self.global_step.eval()
        pickle.dump(config_, info_file)
        info_file.close()
        print("Model saved.")

    def load(self, sess, model_file=None):
        """ Load the model. """
        config = self.config
        if model_file is not None:
            save_path = model_file
##        else:
##            info_path = os.path.join(config.save_dir, "config.pickle")
##            info_file = open(info_path, "rb")
##            config = pickle.load(info_file)
##            global_step = config.global_step
##            info_file.close()
##            save_path = os.path.join(config.save_dir,
##                                     str(global_step)+".npy")

        print("Loading the model from %s..." %save_path)
        data_dict = np.load(save_path,encoding='latin1').item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict.keys():
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print("%d tensors loaded." %count)

    def load_cnn(self, session, data_path, ignore_missing=True):
        """ Load a pretrained CNN model. """
        print("Loading the CNN from %s..." %data_path)
        data_dict = np.load(data_path).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse = True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                    except ValueError:
                        pass
        print("%d tensors loaded." %count)



class CaptionGenerator(BaseModel):
    def build(self):
        """ Build the model. """
        self.build_cnn()
        self.build_rnn()

    def build_cnn(self):
        """ Build the CNN. """
        print("Building the CNN...")
        self.build_vgg16()
        print("CNN built.")

    def build_vgg16(self):
        """ Build the VGG16 net. """
        #config = self.config

        images = tf.placeholder(
            dtype = tf.float32,
            shape = [config.batch_size] + self.image_shape)

        conv1_1_feats = self.nn.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = self.nn.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = self.nn.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = self.nn.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = self.nn.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = self.nn.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = self.nn.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = self.nn.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = self.nn.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = self.nn.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = self.nn.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = self.nn.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = self.nn.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = self.nn.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = self.nn.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = self.nn.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = self.nn.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.batch_size, 196, 512])

        self.conv_feats = reshaped_conv5_3_feats
        self.num_ctx = 196
        self.dim_ctx = 512
        self.images = images

    
    def build_rnn(self):
        """ Build the RNN. """
        print("Building the RNN...")
        config = self.config

        # Setup the placeholders
        contexts = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, self.num_ctx, self.dim_ctx])
        last_memory = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
        last_output = tf.placeholder(
                dtype = tf.float32,
                shape = [config.batch_size, config.num_lstm_units])
        last_word = tf.placeholder(
                dtype = tf.int32,
                shape = [config.batch_size])

        # Setup the word embedding
        with tf.variable_scope("word_embedding"):
            embedding_matrix = tf.get_variable(
                name = 'weights',
                shape = [config.vocabulary_size, config.dim_embedding],
                initializer = self.nn.fc_kernel_initializer,
                regularizer = self.nn.fc_kernel_regularizer,
                trainable = False)

        # Setup the LSTM
        lstm = tf.nn.rnn_cell.LSTMCell(
            config.num_lstm_units,
            initializer = self.nn.fc_kernel_initializer)

        # Initialize the LSTM using the mean context
        with tf.variable_scope("initialize"):
            context_mean = tf.reduce_mean(self.conv_feats, axis = 1)
            initial_memory, initial_output = self.initialize(context_mean)
            initial_state = initial_memory, initial_output

        # Prepare to run
        predictions = []
        num_steps = 1
        last_state = last_memory, last_output

        # Generate the words one by one
        for idx in range(num_steps):
            # Attention mechanism
            with tf.variable_scope("attend"):
                alpha = self.attend(contexts, last_output)
                context = tf.reduce_sum(contexts*tf.expand_dims(alpha, 2),
                                        axis = 1)
        
            # Embed the last word
            with tf.variable_scope("word_embedding"):
                word_embed = tf.nn.embedding_lookup(embedding_matrix,
                                                    last_word)
           # Apply the LSTM
            with tf.variable_scope("lstm"):
                current_input = tf.concat([context, word_embed], 1)
                output, state = lstm(current_input, last_state)
                memory, _ = state

            # Decode the expanded output of LSTM into a word
            with tf.variable_scope("decode"):
                expanded_output = tf.concat([output,
                                             context,
                                             word_embed],
                                             axis = 1)
                logits = self.decode(expanded_output)
                probs = tf.nn.softmax(logits)
                prediction = tf.argmax(logits, 1)
                predictions.append(prediction)

            # Compute the loss for this step, if necessary
            tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
        self.contexts = contexts
        self.initial_memory = initial_memory
        self.initial_output = initial_output
        self.last_memory = last_memory
        self.last_output = last_output
        self.last_word = last_word
        self.memory = memory
        self.output = output
        self.probs = probs

        print("RNN built.")

    def initialize(self, context_mean):
        """ Initialize the LSTM using the mean context. """
        config = self.config
        context_mean = self.nn.dropout(context_mean)
        if config.num_initalize_layers == 1:
            # use 1 fc layer to initialize
            memory = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a')
            output = self.nn.dense(context_mean,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b')
        else:
            # use 2 fc layers to initialize
            temp1 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_a1')
            temp1 = self.nn.dropout(temp1)
            memory = self.nn.dense(temp1,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_a2')

            temp2 = self.nn.dense(context_mean,
                                  units = config.dim_initalize_layer,
                                  activation = tf.tanh,
                                  name = 'fc_b1')
            temp2 = self.nn.dropout(temp2)
            output = self.nn.dense(temp2,
                                   units = config.num_lstm_units,
                                   activation = None,
                                   name = 'fc_b2')
        return memory, output

    def attend(self, contexts, output):
        """ Attention Mechanism. """
        config = self.config
        reshaped_contexts = tf.reshape(contexts, [-1, self.dim_ctx])
        reshaped_contexts = self.nn.dropout(reshaped_contexts)
        output = self.nn.dropout(output)
        if config.num_attend_layers == 1:
            # use 1 fc layer to attend
            logits1 = self.nn.dense(reshaped_contexts,
                                    units = 1,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_a')
            logits1 = tf.reshape(logits1, [-1, self.num_ctx])
            logits2 = self.nn.dense(output,
                                    units = self.num_ctx,
                                    activation = None,
                                    use_bias = False,
                                    name = 'fc_b')
            logits = logits1 + logits2
        else:
            # use 2 fc layers to attend
            temp1 = self.nn.dense(reshaped_contexts,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1a')
            temp2 = self.nn.dense(output,
                                  units = config.dim_attend_layer,
                                  activation = tf.tanh,
                                  name = 'fc_1b')
            temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, self.num_ctx, 1])
            temp2 = tf.reshape(temp2, [-1, config.dim_attend_layer])
            temp = temp1 + temp2
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = 1,
                                   activation = None,
                                   use_bias = False,
                                   name = 'fc_2')
            logits = tf.reshape(logits, [-1, self.num_ctx])
        alpha = tf.nn.softmax(logits)
        return alpha

    def decode(self, expanded_output):
        """ Decode the expanded output of the LSTM into a word. """
        config = self.config
        expanded_output = self.nn.dropout(expanded_output)
        if config.num_decode_layers == 1:
            # use 1 fc layer to decode
            logits = self.nn.dense(expanded_output,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc')
        else:
            # use 2 fc layers to decode
            temp = self.nn.dense(expanded_output,
                                 units = config.dim_decode_layer,
                                 activation = tf.tanh,
                                 name = 'fc_1')
            temp = self.nn.dropout(temp)
            logits = self.nn.dense(temp,
                                   units = config.vocabulary_size,
                                   activation = None,
                                   name = 'fc_2')
        return logits

    def build_optimizer(self):
        """ Setup the optimizer and training operation. """
        config = self.config

        learning_rate = tf.constant(config.initial_learning_rate)
        if config.learning_rate_decay_factor < 1.0:
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps = config.num_steps_per_decay,
                    decay_rate = config.learning_rate_decay_factor,
                    staircase = True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        else:
            learning_rate_decay_fn = None

        with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
            if config.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate = config.initial_learning_rate,
                    beta1 = config.beta1,
                    beta2 = config.beta2,
                    epsilon = config.epsilon
                    )
            else:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = config.initial_learning_rate
                )

            opt_op = tf.contrib.layers.optimize_loss(
                loss = self.total_loss,
                global_step = self.global_step,
                learning_rate = learning_rate,
                optimizer = optimizer,
                clip_gradients = config.clip_gradients,
                learning_rate_decay_fn = learning_rate_decay_fn)

        self.opt_op = opt_op

    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)

        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("attention_loss", self.attention_loss)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("total_loss", self.total_loss)
            tf.summary.scalar("accuracy", self.accuracy)

        with tf.name_scope("attentions"):
            self.variable_summary(self.attentions)

        self.summary = tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class DataSet(object):
    def __init__(self,
                 image_ids,
                 image_files,
                 batch_size,
                 word_idxs=None,
                 masks=None,
                 is_train=False,
                 shuffle=False):
        self.image_ids = np.array(image_ids)
        self.image_files = np.array(image_files)
        self.word_idxs = np.array(word_idxs)
        self.masks = np.array(masks)
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.setup()

    def setup(self):
        """ Setup the dataset. """
        self.count = len(self.image_ids)
        self.num_batches = int(np.ceil(self.count * 1.0 / self.batch_size))
        self.fake_count = self.num_batches * self.batch_size - self.count
        self.idxs = list(range(self.count))
        self.reset()

    def reset(self):
        """ Reset the dataset. """
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        """ Fetch the next batch. """
        assert self.has_next_batch()

        if self.has_full_next_batch():
            start, end = self.current_idx, \
                         self.current_idx + self.batch_size
            current_idxs = self.idxs[start:end]
        else:
            start, end = self.current_idx, self.count
            current_idxs = self.idxs[start:end] + \
                           list(np.random.choice(self.count, self.fake_count))

        image_files = self.image_files[current_idxs]
        if self.is_train:
            word_idxs = self.word_idxs[current_idxs]
            masks = self.masks[current_idxs]
            self.current_idx += self.batch_size
            return image_files, word_idxs, masks
        else:
            self.current_idx += self.batch_size
            return image_files

    def has_next_batch(self):
        """ Determine whether there is a batch left. """
        return self.current_idx < self.count

    def has_full_next_batch(self):
        """ Determine whether there is a full batch left. """
        return self.current_idx + self.batch_size <= self.count







config=Config()

def prepare_test_data(config):
    """ Prepare the data for testing the model. """
    #files = os.listdir(config.test_image_dir)
    #image_files = [os.path.join(config.test_image_dir, f) for f in files
    #    if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    image_files=[config.image_file]
    image_ids = list(range(len(image_files)))

    print("Building the vocabulary...")
    if os.path.exists(config.vocabulary_file):
        vocabulary = Vocabulary(config.vocabulary_size,
                                config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Building the dataset...")
    dataset = DataSet(image_ids, image_files, config.batch_size)
    print("Dataset built.")
    print(type(dataset))
    return dataset,vocabulary


import cv2
import heapq

class ImageLoader(object):
    def __init__(self, mean_file):
        self.bgr = True
        self.scale_shape = np.array([224, 224], np.int32)
        self.crop_shape = np.array([224, 224], np.int32)
        self.mean = np.load(mean_file).mean(1).mean(1)

    def load_image(self, image_file):
        """ Load and preprocess an image. """
        image = cv2.imread(image_file)

        if self.bgr:
            temp = image.swapaxes(0, 2)
            temp = temp[::-1]
            image = temp.swapaxes(0, 2)

        image = cv2.resize(image, (self.scale_shape[0], self.scale_shape[1]))
        offset = (self.scale_shape - self.crop_shape) / 2
        offset = offset.astype(np.int32)
        image = image[offset[0]:offset[0]+self.crop_shape[0],
                      offset[1]:offset[1]+self.crop_shape[1]]
        image = image - self.mean
        return image

    def load_images(self, image_files):
        """ Load and preprocess a list of images. """
        images = []
        for image_file in image_files:
            images.append(self.load_image(image_file))
        images = np.array(images, np.float32)
        return images

class CaptionData(object):
    def __init__(self, sentence, memory, output, score):
       self.sentence = sentence
       self.memory = memory
       self.output = output
       self.score = score

    def __cmp__(self, other):
        assert isinstance(other, CaptionData)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, CaptionData)
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, CaptionData)
        return self.score == other.score

class TopN(object):
    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        self._data = []


        

class Vocabulary(object):
    def __init__(self, size, save_file=None):
        self.words = []
        self.word2idx = {}
        self.word_frequencies = []
        self.size = size
        if save_file is not None:
            self.load(save_file)

    def build(self, sentences):
        """ Build the vocabulary and compute the frequency of each word. """
        word_counts = {}
        for sentence in tqdm(sentences):
            for w in word_tokenize(sentence.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1.0

        assert self.size-1 <= len(word_counts.keys())
        self.words.append('<start>')
        self.word2idx['<start>'] = 0
        self.word_frequencies.append(1.0)

        word_counts = sorted(list(word_counts.items()),
                            key=lambda x: x[1],
                            reverse=True)

        for idx in range(self.size-1):
            word, frequency = word_counts[idx]
            self.words.append(word)
            self.word2idx[word] = idx + 1
            self.word_frequencies.append(frequency)

        self.word_frequencies = np.array(self.word_frequencies)
        self.word_frequencies /= np.sum(self.word_frequencies)
        self.word_frequencies = np.log(self.word_frequencies)
        self.word_frequencies -= np.max(self.word_frequencies)

    def process_sentence(self, sentence):
        """ Tokenize a sentence, and translate each token into its index
            in the vocabulary. """
        words = word_tokenize(sentence.lower())
        word_idxs = [self.word2idx[w] for w in words]
        return word_idxs

    def get_sentence(self, idxs):
        """ Translate a vector of indicies into a sentence. """
        words = [self.words[i] for i in idxs]
        if words[-1] != '.':
            words.append('.')
        length = np.argmax(np.array(words)=='.') + 1
        words = words[:length]
        sentence = "".join([" "+w if not w.startswith("'") \
                            and w not in string.punctuation \
                            else w for w in words]).strip()
        return sentence

    def save(self, save_file):
        """ Save the vocabulary to a file. """
        data = pd.DataFrame({'word': self.words,
                             'index': list(range(self.size)),
                             'frequency': self.word_frequencies})
        data.to_csv(save_file)

    def load(self,save_file):
        """ Load the vocabulary from a file. """
        assert os.path.exists(save_file)
        data = pd.read_csv(save_file)
        self.words = data['word'].values
        self.word2idx = {self.words[i]:i for i in range(self.size)}
        self.word_frequencies = data['frequency'].values




def main(argv):
    config = Config()
    with tf.Session() as sess:
        data, vocabulary = prepare_test_data(config)
        model = CaptionGenerator(config)
        model.load(sess,config.model_file)
        tf.get_default_graph().finalize()
        model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()
