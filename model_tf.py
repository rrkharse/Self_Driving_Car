# Import sklearn function just for train_test_split
from sklearn.model_selection import train_test_split
# Import sklearn function just for shuffle function 
# to randomnly select x,y values for each batch when training
from sklearn.utils import shuffle
# Read images
import matplotlib.image as mpimg

import tensorflow as tf
import sys
import pandas as pd 
import numpy as np 
import argparse
import os

# helper class
import utils_tf

args = None

def main(_):
	X_train, X_valid, y_train, y_valid = load_data(args)
	train(X_train, X_valid, y_train, y_valid,args)


def parse_args():
        parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
        parser.register("type", "bool", lambda v: v.lower() == "true")
        parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
        parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
        parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
        parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=100)
        parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=2000)
        parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
        parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
        parser.add_argument('-N', help='model name',         dest='model_name',     type=str, default='epochs/model')
        parser.add_argument(
            "--ps_hosts",
            type=str,
            default="",
            help="Comma-separated list of hostname:port pairs"
        )
        parser.add_argument(
            "--worker_hosts",
            type=str,
            default="",
            help="Comma-separated list of hostname:port pairs"
        )
        parser.add_argument(
            "--job_name",
            type=str,
            default="",
            help="One of 'ps', 'worker'"
        )
        # Flags for defining the tf.train.Server
        parser.add_argument(
            "--task_index",
            type=int,
            default=0,
            help="Index of task within the job"
        )
        flags, unp = parser.parse_known_args()
        return flags, unp



def load_data(args):

	# Read all data from csv
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # Take relevant columns from data
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    # Run split 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    # Run utils_tf tool to generate varied data (shadows, discoloration, etc.)
    X_train_plus, y_train_plus = utils_tf.gen_double_set(X_train,y_train)

    # Store testing data as numpy format
    X_valid = np.array([utils_tf.preprocess(mpimg.imread(i[0])) for i in X_valid])
    y_valid = np.array([[i] for i in y_valid])

    return X_train_plus, X_valid, y_train_plus, y_valid


def train(X_train, X_valid, y_train, y_valid, args):

        ps_hosts = args.ps_hosts.split(",")
        worker_hosts = args.worker_hosts.split(",")

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                           job_name=args.job_name,
                           task_index=args.task_index)


        if args.job_name == "ps":
            server.join()
        elif args.job_name == "worker":

            with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % args.task_index,
                cluster=cluster)):
            
                x = tf.placeholder(tf.float32,[None,160,320,3],name="x")
                y_real = tf.placeholder(tf.float32,[None,1])
                
                y_conv, keep_prob = model_scheme(x)
                
                cost_function = tf.reduce_mean(tf.square(y_conv - y_real))
                global_step = tf.contrib.framework.get_or_create_global_step()
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_function, global_step=global_step)
                
                
                hooks=[tf.train.StopAtStepHook(last_step=args.nb_epoch*args.samples_per_epoch)]
                
                with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(args.task_index == 0), checkpoint_dir="checkpoints",
                                           hooks=hooks) as sess:
                	#sess.run(tf.global_variables_initializer())	
                        while not sess.should_stop():
                            b_sz = args.batch_size
                            X_train, y_train = shuffle(X_train,y_train,random_state=0)
                            feed_dict={x:X_train[:b_sz],y_real:y_train[:b_sz],keep_prob: args.keep_prob}
                            sess.run(train_step, feed_dict=feed_dict)
                            
                            if global_step % 100 == 0:
                                feed_dict={x:X_train[:b_sz],y_real:y_train[:b_sz], keep_prob: 1.0}
                                difference = tf.sqrt(cost_function).eval(feed_dict=feed_dict)
                                print('step %d, difference %g'%(global_step,difference))
                                
                            #if global_step % args.samples_per_epoch == 0:
                            #    saver = tf.train.Saver() 
                            #    saver.save(sess,'./{}{}'.format(args.model_name,str(global_step/args.samples_per_epoch)),global_step=global_step)



def model_scheme(x):

 #    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
 #    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
 #    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
 #    model.add(Conv2D(64, 3, 3, activation='elu'))
 #    model.add(Conv2D(64, 3, 3, activation='elu'))
 #    model.add(Dropout(args.keep_prob))
 #    model.add(Flatten())
 #    model.add(Dense(100, activation='elu'))
 #    model.add(Dense(50, activation='elu'))
 #    model.add(Dense(10, activation='elu'))
 #    model.add(Dense(1))

	x_image = tf.reshape(tf.cast(x,tf.float32),[-1,160,320,3]) / tf.constant(127.5) - tf.constant(1.0)

	y_pool1 = conv_pool(x_image,5,3,24)
	y_pool2 = conv_pool(y_pool1,5,24,36)
	y_pool3 = conv_pool(y_pool2,5,36,48)
	y_pool4 = conv_pool(y_pool3,3,48,64)
	y_pool5 = conv_pool(y_pool4,3,64,64)

	keep_prob = tf.placeholder(tf.float32,name="keep_prob")
	y_drop = tf.nn.dropout(y_pool5,keep_prob)

	y_flat = tf.reshape(y_drop,[-1,10*5*64])

	y_fc1 = fc_mat_mul(y_flat,10*5*64,100)
	y_fc2 = fc_mat_mul(y_fc1,100,50)
	y_fc3 = fc_mat_mul(y_fc2,50,10)
	y_out = fc_mat_mul(y_fc3,10,1)

	y_conv_f = tf.add(y_out,tf.constant(0.0),name="op_to_restore")

	return y_conv_f, keep_prob


def conv_pool(x,conv_val,feat_in_size,feat_out_size):
	"""
	Function to apply convolution and pooling
	"""
	W = weight_variable([conv_val,conv_val,feat_in_size,feat_out_size])
	b = bias_variable([feat_out_size])
	y = tf.nn.elu(conv2d(x,W)+b)
	y_pool = max_pool_2x2(y)

	print(y,y_pool)
	return y_pool


def fc_mat_mul(x,height,width):
	"""
	Function to multiply matrices between fully connected layers
	"""
	W = weight_variable([height,width])
	b = bias_variable([width])
	y = tf.nn.elu(tf.matmul(x,W)+b)

	print(y)
	return y


def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))	#Returns weight variable initialized with random value

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))	#Returns bias variable initialized with random value

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')	# Runs convolution operation on x input using W weights

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')	# Pools input x to downsample data


if __name__ == "__main__":
        args, unparsed = parse_args()
        tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
