# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:10:40 2018

@author: edwba
"""

import tensorflow as tf
import numpy as np
#import os

## CREATE NN ARCHITECTURE

## PERMITS TENSORFLOW EVAL AND TRAINING

class nn:
    
    ZX = 0 # declare only (will fail if not initialised)
    ZY = 0 # declare only (will fail if not initialised)
    A = 0
    
    nW1 = 100
    nW2 = 20
    stddev = 0.01
    dropout = 0.5 # proportion to keep
    
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    D = tf.placeholder('float', [], name = 'D') # for temporal difference delta
    OA = tf.placeholder('int32', [], name = 'OA') # for the action selected (to compute derivative)
        
    def __init__(self, ZX, ZY):
        
        self.ZX = ZX
        self.ZY = ZY
        
        print("\nTotal weights: ", ZX * self.nW1 + self.nW1 * self.nW2 + self.nW2 * ZY,"\n")
        
        # basic architecture
        
        self.X = tf.placeholder('float', [1, ZX], name = 'X') # for input vector

        self.W1 = tf.Variable(tf.random_normal([ZX, self.nW1], stddev=self.stddev), name = 'W1')
        self.W1do = tf.nn.dropout(self.W1,self.dropout)
        self.L1 = tf.nn.relu(tf.matmul(self.X, self.W1do), name = 'L1')
        
        self.W2 = tf.Variable(tf.random_normal([self.nW1, self.nW2], stddev=self.stddev), name = 'W2')
        self.W2do = tf.nn.dropout(self.W2,self.dropout)
        self.L2 = tf.nn.sigmoid(tf.matmul(self.L1, self.W2do), name = 'L2')
        
        self.W3 = tf.Variable(tf.random_normal([self.nW2, ZY], stddev=self.stddev), name = 'W3')
        
        self.Q = tf.matmul(self.L2, self.W3, name = 'Q')
        
        self.nambla1 = tf.gradients(tf.slice(tf.reshape(self.Q,[ZY]), [self.OA], [1]), self.W1)[0]
        self.nambla2 = tf.gradients(tf.slice(tf.reshape(self.Q,[ZY]), [self.OA], [1]), self.W2)[0]
        self.nambla3 = tf.gradients(tf.slice(tf.reshape(self.Q,[ZY]), [self.OA], [1]), self.W3)[0]
        
        self.delW1 = tf.multiply(self.D, self.nambla1)
        self.delW2 = tf.multiply(self.D, self.nambla2)
        self.delW3 = tf.multiply(self.D, self.nambla3)
        
        self.incrW1 = tf.assign(self.W1, tf.add(self.W1, self.delW1))
        self.incrW2 = tf.assign(self.W2, tf.add(self.W2, self.delW2))
        self.incrW3 = tf.assign(self.W3, tf.add(self.W3, self.delW3))
        
    def initialise_nn(self):
        
        self.session.run(tf.global_variables_initializer())
        
    def output_nn(self, x):
        return self.session.run(self.Q, feed_dict={self.X: np.reshape(x,[1,self.ZX])}).flatten()
    
    def update_nn(self, x_old, a_old, delta):
        self.session.run([self.incrW1, self.incrW2, self.incrW3], feed_dict={self.X: np.reshape(x_old,[1,self.ZX]), self.OA: a_old - 1, self.D: delta})

    def close_nn(self):
        self.session.close()
        
    def get_architecture_nn(self):
        return [[self.ZX, self.nW1], [self.nW1, self.nW2], [self.nW2, self.nW3]]
    
    def get_weights_nn(self):
        return [self.W1.eval(session=self.session), self.W2.eval(session=self.session), self.W3.eval(session=self.session)]
    
    