import os
import time
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from nets import preprocessing, SpinePathNet,losses 
import tf_utils
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

Fold = 1

tf.app.flags.DEFINE_string(
    'train_dir', 'tmp/tfmodels_gan_lstm_%s/'%Fold,
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1, 'GPU memory fraction to use.')

tf.app.flags.DEFINE_float(
    'weight', 1, 'The weight is between cross entropy loss and metric loss.')
tf.app.flags.DEFINE_string(
    'dataset_name', 'spine_segmentation', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 7, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test_%s_fold'%Fold, 'The name of the train/test split.')#spine_segmentation_train_7_class.tfrecord
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/tfrecords_spine_segmentation_with_superpixels/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'num_samples', 202, 'The number of samples in the training set.')
tf.app.flags.DEFINE_integer(
    'num_readers', 20,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.94,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float( 
    'num_epochs_per_decay', 10.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('num_epochs', 500,
                            'The number of training epochs.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'tmp/tfmodels_gan_lstm_%s/'%Fold,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', None,#['SpinePathNet/feature_embedding','SpinePathNet/logits/'],
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS



def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    with tf.Graph().as_default():
        with tf.device('CPU:0'):
            global_step = tf.train.create_global_step()
        
        dataset = FLAGS.dataset_dir + '%s_%s.tfrecord' %(FLAGS.dataset_name,FLAGS.dataset_split_name) 
        
        with tf.device('CPU:0'):
            with tf.name_scope('input'):  
                filename_queue = tf.train.string_input_producer([dataset], num_epochs=FLAGS.num_epochs)
            
                image,mask_class,_ = tf_utils.read_and_decode_for_lstm(filename_queue, batch_size = FLAGS.batch_size,\
                                                                       capacity=20 * FLAGS.batch_size,\
                                                                       num_threads=FLAGS.num_readers,\
                                                                       min_after_dequeue=10 * FLAGS.batch_size, is_training=True)
                labels = tf.to_float(tf.contrib.layers.one_hot_encoding(mask_class,FLAGS.num_classes))
                mask_class_onehot_for_d = tf.to_float(tf.contrib.layers.one_hot_encoding(mask_class,FLAGS.num_classes, on_value=0.99,off_value=0.01))  
                labels = tf.reshape(labels, (-1, FLAGS.num_classes))
            
            image = preprocessing.data_augmentation(image, is_training=True)

        #image,mask_class,mask_instance = preprocessing(image,mask_class,mask_instance,is_train=True,data_format= 'NHWC')                   
        logits,_,_= SpinePathNet.g_l_net(image, batch_size=FLAGS.batch_size, class_num=FLAGS.num_classes, reuse=tf.AUTO_REUSE, is_training=True, scope='g_SpinePathNet')           
        D_logit_real = SpinePathNet.d_net(mask_class_onehot_for_d,  class_num = FLAGS.num_classes, reuse=None, is_training=True, scope='d_gan')
        D_logit_fake = SpinePathNet.d_net(logits,  class_num = FLAGS.num_classes, reuse=True, is_training=True, scope='d_gan') 
        
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy_loss = losses.weighted_cross_entropy_with_logits(FLAGS, Fold, logits, labels)
            
        with tf.name_scope('gan_loss'): 
            D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
            D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
            G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
            
            #lmada = tf.Variable(2., name = 'weight_of_gan')
            
            D_loss = D_loss_real + D_loss_fake
        
            G_loss = cross_entropy_loss + G_loss_fake
   
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
       # print (g_vars)
        
        learning_rate = tf_utils.configure_learning_rate(FLAGS,FLAGS.num_samples, global_step)
        
        
        optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)
        optimizer_gan = tf.train.AdamOptimizer(beta1=0.5, learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        #Note: when training, the moving_mean and moving_variance need to be updated.
        with tf.control_dependencies(update_ops):
            train_op_G = optimizer.minimize(G_loss, global_step=global_step, var_list=g_vars)            
            #train_op_fake = optimizer.minimize(G_loss_fake, var_list=g_vars)            
            train_op_D = optimizer_gan.minimize(D_loss, global_step=global_step, var_list=d_vars)
                                                                      
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        sess = tf.Session()
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)

        if ckpt and ckpt.model_checkpoint_path:
            exclude_list = FLAGS.ignore_missing_vars
            variables_list  = tf.contrib.framework.get_variables_to_restore(exclude=exclude_list)
            restore = tf.train.Saver(variables_list)
            restore.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = int(global_step)

        else: global_step = 0    
 
        if not tf.gfile.Exists(FLAGS.train_dir):
                #tf.gfile.DeleteRecursively(FLAGS.train_dir)
                tf.io.gfile.MakeDirs(FLAGS.train_dir)
                
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        
        try:
            step = global_step 
            start_time = time.time()
               
            for i in range(50):
                print("step=", i)
                _, dl, dlr, dlf = sess.run([train_op_D, D_loss, D_loss_real, D_loss_fake]) 
                #cl = sess.run(cross_entropy_loss)
                print('Step %d: cross entropy loss = %.2f' % (step, cl))
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', cl, step=step)
                    
                _,gl, cel, fake_loss,  lr = sess.run([train_op_G, G_loss, cross_entropy_loss, G_loss_fake, learning_rate])           
                
                duration = time.time() - start_time
                
                if step % 10 == 0:
                    print('Step %d: All generative loss = %.2f (Cross entropy loss = %.2f, Fake loss of generatation = %.2f); All discrimator loss = %.2f (Discrimator loss of real = %.2f, Discrimator loss of fake = %.2f); Learning rate = %.4f (%.3f sec)' % (step, gl, cel, fake_loss, dl, dlr, dlf, lr, duration))
                    
                step += 1
                if step % 1000 == 0: #or (step + 1) == FLAGS.max_steps
                    #lmada.assign_add(0.1)
                    saver.save(sess, checkpoint_path, global_step=step)
                   
                                                                      
        except tf.errors.OutOfRangeError:
                
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))  
            saver.save(sess, checkpoint_path, global_step=step) 
            print('Model is saved as %s') % checkpoint_path 

        sess.close()                                                      
                                                                         

if __name__ == '__main__':
    tf.app.run()    