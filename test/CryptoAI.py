from __future__ import print_function
import numpy as np
import random
import pandas as pd
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import math
import itertools
import threading
from matplotlib import pyplot as plt
import tensorflow as tf
import re
import time
import pickle
import datetime

# Utility Function to return True / False regex matching
def pattern_match(patt, string):
    return re.findall(patt, string) != []
# Utility Function to save objects in memory to a file
def save_memory(obj, path):
    return pickle.dump(obj, open(path, "wb"))
# Utility Function to load objects from the harddisk
def load_memory(path):
    return pickle.load(open(path, "rb"))

#--------------------------------------------------------------------------------------
# Read in the price data
#--------------------------------------------------------------------------------------
print("Loading Data...", end="")
# == HACK JOB FOR TESTING SAVE AND TENSORBOARD - start-0 == #
data_raw = pd.read_csv("M30/ALL.csv").dropna(axis=0, how='any').reset_index(drop=True)
# == HACK JOB FOR TESTING SAVE AND TENSORBOARD - end-0 == #
data     = data_raw.drop('date', axis=1)
data['reward_USD'] = 0
# == HACK JOB FOR TESTING SAVE AND TENSORBOARD - start-1 == #
data     = data[:5000]  # Sub-set for testing
# == HACK JOB FOR TESTING SAVE AND TENSORBOARD - end-1 == #
print("{} rows & {} columns".format(len(data), len(data.columns)))
#--------------------------------------------------------------------------------------
# Manual Options
#--------------------------------------------------------------------------------------
COMMISSION     = 0.003      # Commision % as a decimal to use in loss function
USE_PCA        = False       # Use PCA Dimensionality Reduction
PCA_COMPONENTS = 400        # Number of Principle Components to reduce down to
USE_SUPER      = False      # Create new features using supervised learning
INCLUDE_VOLUME = True       # Include Volume as a feature
ALLOW_SHORTS   = False      # Allow Shorts or not
DISCOUNT       = True       # Train on discounted rewards
DISCOUNT_STEPS = 24         # Number of periods to look ahead for discounting
GAMMA          = 0.25       # The discount factor
EPOCH_CNT      = 10000      # Number of epochs for training
LOSSCHK_CNT    = 500        # Check for loss every this number of training epochs
SAVE_MODELS    = True
SAVE_LENGTH    = 0.33       # Save all pre-processing models from this percentage of raw data onwards
SAVE_PATH      = "saved"    # Where teh models get saved
TRADING_PATH   = "Live Trading"
LOG_PATH       = "logs"     # Where to save Tensorboard output
TODAY          = str(datetime.date.today()) # Todays date
#--------------------------------------------------------------------------------------
# Defining the batch size and test length
#--------------------------------------------------------------------------------------
BATCH_SZ_MIN = 100
BATCH_SZ_MAX = 200
TEST_LEN     = int(round(0.2*len(data)))
IDX_MAX      = int(max(0, len(data) - TEST_LEN - BATCH_SZ_MAX - 1))
SAVE_IDX     = int(round(SAVE_LENGTH * len(data_raw)))
#--------------------------------------------------------------------------------------
# List of coins to trade. Set to [] to use all coins
#--------------------------------------------------------------------------------------
COINS       = ['USD', 'BCH', 'BTC', 'DASH', 'ETC', 'ETH', 'LTC', 'XMR', 'XRP', 'ZEC']
# List of coins data to use as input variables. Set to [] to use all coins
#--------------------------------------------------------------------------------------
INPUT_COINS = []
N_VEC       = 3 + 1 if INCLUDE_VOLUME else 0
N_COINS     = ( len(COINS) * 2 - 1 ) if ALLOW_SHORTS else len(COINS)
#--------------------------------------------------------------------------------------
# Create output directories
#--------------------------------------------------------------------------------------
for path in ([TRADING_PATH,LOG_PATH,SAVE_PATH]):
    try:
        os.mkdir(path)
    except FileExistsError as e1:
        pass
    except OSError as e2:
        print('Failed to create directory {} - Incorrect syntax?'.format(path))
    except:
        print('Error occurred - {}.'.format(sys.exc_info()[0]))                        
#--------------------------------------------------------------------------------------
# Create fields to store "Previous Weights" - Only needed when commission is > 0
#--------------------------------------------------------------------------------------
PORT_W = []
if COMMISSION > 0:
    PORT_W.append('MARGIN_USD')
    data["MARGIN_USD"] = 1
    for i, a in enumerate(sorted(COINS)):
        data["MARGIN_{}".format(a)] = 1 if a == "USD" else 0
        if "MARGIN_{}".format(a) not in PORT_W:
            PORT_W.append("MARGIN_{}".format(a))
    if ALLOW_SHORTS:
        for i, a in enumerate(sorted(COINS)):
            if a in ["USD", "USDT"]:
                continue
            data["MARGIN_{}_S".format(a)] = 0

if ALLOW_SHORTS:
    x = list(PORT_W)
    for asset in x[1:]:
        PORT_W.append(asset+"_S")
#--------------------------------------------------------------------------------------
# Create a list of X column names to use for modelling
#--------------------------------------------------------------------------------------
in_cols = []
for c in data.columns:
    if INPUT_COINS == []:
        in_cols.append(c)
    else:
        for a in set(INPUT_COINS):
            if a in c:
                in_cols.append(c)

COLS_X = []
for x in in_cols:
    if "L_" in x or "REG" in x:
        if "VOLUME" in x and INCLUDE_VOLUME == False:
            continue
        COLS_X.append(x)
if COMMISSION != 0:
    COLS_X += PORT_W
#--------------------------------------------------------------------------------------
# Create a list of Y column names to use for modelling
#--------------------------------------------------------------------------------------

COLS_Y = ["reward_USD"]
for c in data.columns:
    added = False
    if 'reward' in c and 'USD' not in c:
        if COINS == []:
            COLS_Y += [c]
            added = True
        else:
            for a in set(COINS):
                if a in c:
                    COLS_Y += [c]
                    added = True
        if added:
            data[c+"_S"] = data[c].apply(lambda x : math.log10(2-10**x))
if ALLOW_SHORTS:
    COLS_Y += ["{}_S".format(y) for y in COLS_Y[1:]]
#--------------------------------------------------------------------------------------
# Normalizing the X columns. Scale using training data only
#--------------------------------------------------------------------------------------
print("Normalizing Data...", end="")
for x in COLS_X:
    median      = data[SAVE_IDX:][x].describe()[5]
    data[x]     = data[x].apply(lambda x : median if np.isinf(x) or np.isnan(x) else x)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit( data[:IDX_MAX+BATCH_SZ_MAX] [COLS_X] )
data[COLS_X] = scaler.transform(data[COLS_X])
if SAVE_MODELS:
    live_scaler = sklearn.preprocessing.StandardScaler()
    live_scaler.fit( data[SAVE_IDX:] [COLS_X] )
    save_memory(live_scaler, TRADING_PATH+"/Scaler.save")
    save_memory(COLS_X, TRADING_PATH+"/COLS_X_ORIG.save")
print("Done")
#--------------------------------------------------------------------------------------
# Apply PCA if set to True. Principle Components calculated using training data only
#--------------------------------------------------------------------------------------
if USE_PCA:

    print("PCA...",end="")
    PCA_MODEL = sklearn.decomposition.PCA(PCA_COMPONENTS)
    PCA_MODEL.fit(data[:IDX_MAX+BATCH_SZ_MAX][COLS_X])
    Xs = pd.DataFrame(PCA_MODEL.transform(data[COLS_X]))
    
    Xs.columns = ["PCA_"+str(x) for x in range(1,len(Xs.columns)+1)]
    data[Xs.columns] = Xs
    COLS_X = list(Xs.columns) + (PORT_W if COMMISSION != 0 else [])
    
    if SAVE_MODELS:
        live_pca = sklearn.decomposition.PCA(PCA_COMPONENTS)
        live_pca.fit( data[SAVE_IDX:] [COLS_X] )
        save_memory(live_pca, TRADING_PATH+"/PCA.save")

#    print(PCA_MODEL.explained_variance_)
#    print(PCA_MODEL.explained_variance_ratio_)
    print("Done")
    print("Variance explained: {}".format(100*PCA_MODEL.explained_variance_ratio_.cumsum()[-1]))
#--------------------------------------------------------------------------------------
# Generate Supervised Learning Predictions if set to True. This does not work for now
#--------------------------------------------------------------------------------------
if USE_SUPER:
    pass
    '''training = data[:IDX_MAX]
    cols_to_add = []
    for target in COLS_Y:
        model = sklearn.ensemble.RandomForestRegressor()
        model.fit(training[COLS_X], training[target])
        newcol = "RF_{}".format(target)
        data[newcol] = model.predict(data[COLS_X])
        cols_to_add.append(newcol)
    COLS_X += cols_to_add'''
    
#--------------------------------------------------------------------------------------
# Transform rewards into discounted reward if enabled. "data" uses transformed 
# rewards, "data_imm" uses raw, un-modified reward.
#--------------------------------------------------------------------------------------
if DISCOUNT:
    data_imm = data.copy()
    stmt = "data[COLS_Y] = data[COLS_Y]"
    for ahead in range(1,DISCOUNT_STEPS+1):
        stmt += "+(GAMMA**{}) * data[COLS_Y].shift({})".format(ahead, -ahead)
    print("Calculating Discount Rewards...", end="")
    exec(stmt)
    print("Done")
else:
    data_imm = data.copy()

data = data.dropna(axis=0, how='any').reset_index(drop=True)
if DISCOUNT:
    data_imm = data_imm[:-DISCOUNT_STEPS]
    
'''for c in COLS_Y:
    data[c] = data[c] - math.log10(1.004)
data["reward_USD"] = 0'''
    
N_IN  = len(COLS_X)
N_OUT = len(COLS_Y)

if SAVE_MODELS:
    save_memory(COLS_X, TRADING_PATH+"/COLS_X.save")
    save_memory(COLS_Y, TRADING_PATH+"/COLS_Y.save")

#--------------------------------------------------------------------------------------
#  
#                                NEURAL NETWORK DESIGN
#
#--------------------------------------------------------------------------------------

# Input / Output place holders
X = tf.placeholder(tf.float32, [None, N_IN])
X = tf.reshape(X, [-1, N_IN])
# PrevW
PREV_W = tf.placeholder(tf.float32, [None, N_OUT])
# Actual Rewards
Y_     = tf.placeholder(tf.float32, [None, N_OUT])
#--------------------------------------------------------------------------------------
# Define hidden layers
#--------------------------------------------------------------------------------------
# Define number of Neurons per layer
K = 100 # Layer 1
L = 100 # Layer 2
M = 100 # Layer 3
N = 100 # Layer 4

SDEV = 0.1

# LAYER 1
W1 = tf.Variable(tf.random_normal([N_IN, K], stddev = SDEV))
B1 = tf.Variable(tf.random_normal([K], stddev = SDEV))

# LAYER 2
W2 = tf.Variable(tf.random_normal([K, L], stddev = SDEV))
B2 = tf.Variable(tf.random_normal([L], stddev = SDEV))

# LAYER 3
W3 = tf.Variable(tf.random_normal([L, M], stddev = SDEV))
B3 = tf.Variable(tf.random_normal([M], stddev = SDEV))

# LAYER 4
W4 = tf.Variable(tf.random_normal([M, N_OUT], stddev = SDEV))
B4 = tf.Variable(tf.random_normal([N_OUT], stddev = SDEV))

reg_losses =  tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
reg_losses += tf.nn.l2_loss(B1) + tf.nn.l2_loss(B2) + tf.nn.l2_loss(B3)

# Magic number is around 0.0001
lambda_reg = 0.0001

#--------------------------------------------------------------------------------------
# Define Computation Graph
#--------------------------------------------------------------------------------------
# Feed forward. Output of previous is input to next
# Activation function for final layer is Softmax for portfolio weights in the range [0,1]
H1  = tf.nn.relu(tf.matmul(X,  W1) + B1)
DH1 = tf.nn.dropout(H1, 0.9)
H2  = tf.nn.relu(tf.matmul(DH1, W2) + B2)
DH2 = tf.nn.dropout(H2, 0.9)
H3  = tf.nn.relu(tf.matmul(DH2, W3) + B3)
DH3 = tf.nn.dropout(H2, 0.9)
Y   = tf.nn.softmax(tf.matmul(DH3, W4) + B4)
#Y_MAX = tf.sign(Y - tf.reduce_max(Y,axis=1,keep_dims=True)) + 1
#--------------------------------------------------------------------------------------
# Define Loss Function
#--------------------------------------------------------------------------------------
if COMMISSION == 0:
    weight_moves = tf.reduce_mean(tf.reduce_sum(tf.abs(Y[1:] - Y[:-1]), axis=1))
    tensor_rwds = tf.log (10**tf.reduce_sum(Y * Y_, axis=1) )
    reward      = tf.reduce_sum(tensor_rwds)
    loss        = -tf.reduce_mean( tensor_rwds ) + tf.log(weight_moves)*0.0002 + lambda_reg * reg_losses
else:
    weight_moves = tf.reduce_mean(tf.reduce_sum(tf.abs(Y[1:] - Y[:-1]), axis=1))
    tensor_rwds = tf.log (tf.reduce_sum( ( 1-COMMISSION*tf.abs(Y-PREV_W) ) * (Y * 10**Y_), axis=1))
    reward      = tf.reduce_sum( tensor_rwds )
    loss        = -tf.reduce_mean( tensor_rwds ) + lambda_reg * reg_losses

# === Tensorboard - start-0 === #
# Create a summary to monitor cost tensor
tf.summary.scalar("loss",loss)
# Create a summary to monitor rewards tensor
tf.summary.scalar("reward",reward)
#tf.summary.scalar("tensor_rwds",[tensor_rwds])
# Create a summary to monitor rewards tensor
tf.summary.scalar("weight_moves",weight_moves)
# Merge all summaries into a single op
summary_op = tf.summary.merge_all()
# === Tensorboard - end-0 === #

# Optimizer
LEARNING_RATE 	= 0.0002
optimizer 	= tf.train.AdamOptimizer(LEARNING_RATE)
train_step 	= optimizer.minimize(loss)

test_imm   = data_imm.iloc[len(data_imm)-TEST_LEN:, :].reset_index(drop=True)
test_dat   = data.iloc[len(data)-TEST_LEN:, :].reset_index(drop=True)

feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT))}
                                   
feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1,N_IN)), 
            Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT))}

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# === Tensorboard - start-1 === #
# op to write logs to Tensorboard
writer = tf.summary.FileWriter(LOG_PATH,graph=tf.get_default_graph()) #sess.graph ?
# === Tensorboard - end-1 === #

# == Save Model - start-0 == #
#@URL: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
if SAVE_MODELS:
    #saves a model every 1 hour and maximum 4 latest models are saved.
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=1)
    saver.save(sess, '{}'.format(SAVE_PATH + os.sep + 'model_out_' + TODAY))
# == Save Model - end-0 == #

dat_rwds, imm_rwds, dat_losses, imm_losses = [], [], [], []
print("Begin Learning...")
#---------------------------------------------------------------------------------------------------
for epoch in range(EPOCH_CNT):
    
    # Measure loss on validation set every LOSSCHK_CNT epochs
    if epoch % LOSSCHK_CNT == 0:
        
        if COMMISSION != 0:
            prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
            stime = time.time()
            test_dat.at[0,PORT_W] = prev_weights[-1]
            test_imm.at[0,PORT_W] = prev_weights[-1]
            b_x = np.reshape(test_dat[COLS_X], (-1,N_IN))
            b_y = np.reshape(test_dat[COLS_Y], (-1,N_OUT))
            #---------------------------------------------------------
            for r in range(len(test_dat) - 1):
                feed_row = {X:  np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
                                               
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                y_vec_10 = (weights * 10 ** y_vec)
                w = y_vec_10 / np.sum(y_vec_10)
                prev_weights.append(w[0])
                
                b_x.at[r+1,PORT_W]      = w[0]
                test_dat.at[r+1,PORT_W] = w[0]
                test_imm.at[r+1,PORT_W] = w[0]
                #print(r / (time.time() - stime))
            #---------------------------------------------------------
                
            feed_dat = {X:  np.reshape(test_dat[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_dat[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                                   
            feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                        Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                        PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
                
        d_rwd, d_loss = sess.run([reward, loss], feed_dict=feed_dat)
        i_rwd, i_loss = sess.run([reward, loss], feed_dict=feed_imm)
        
        dat_rwds.append(math.exp(d_rwd))
        imm_rwds.append(math.exp(i_rwd))
        dat_losses.append(d_loss)
        imm_losses.append(i_loss)
        print("Epoch {:<9} Loss: {:<12.6f} {:<12.6f} Reward: {:<12.6f} {:<12.6f}".format(
                epoch
                ,dat_losses[-1]
                ,dat_losses[-1]
                ,dat_rwds[-1]
                ,imm_rwds[-1])
        )

    #-----------------------------------------------------------------
        
    idx      = int(round(random.random()**0.8*IDX_MAX))
    batch_sz = random.randint(BATCH_SZ_MIN, BATCH_SZ_MAX)
    sub_data = data.iloc[idx:idx+batch_sz, :].reset_index(drop=True)
    batch_X, batch_Y = (sub_data[COLS_X], sub_data[COLS_Y])
    
    if COMMISSION != 0:
        prev_weights = []
        rand = np.random.random(N_OUT)
        rand /= rand.sum()
        prev_weights.append(list(rand))
        batch_X.at[0,PORT_W] = prev_weights
        b_x = np.reshape(batch_X, (-1,N_IN))
        b_y = np.reshape(batch_Y, (-1,N_OUT))
        for r in range(len(batch_X) - 1):
            if random.random() < 0.03 or True:
                rand = np.random.random(N_OUT)
                rand /= rand.sum()
                prev_weights.append(rand)
                b_x.at[r+1,PORT_W] = rand
            else:
                feed_row = {X: np.reshape(np.array(b_x.iloc[r,:]), (-1,N_IN)),
                            Y_: np.reshape(np.array(b_y.iloc[r,:]), (-1,N_OUT)),
                            PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
                weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
                w = (weights * 10** y_vec) / np.sum(weights * 10 ** y_vec)
                prev_weights.append(w[0])
                b_x.at[r+1,PORT_W] = w[0]
                
        batch_X = b_x
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT)),
                      PREV_W: np.reshape(prev_weights, (-1, N_OUT))}
        
    else:
        train_data = {X:  np.reshape(batch_X, (-1,N_IN)), 
                      Y_: np.reshape(batch_Y, (-1,N_OUT))}
        
    step = sess.run(train_step, feed_dict=train_data)
#    a_rwd = sess.run(reward, feed_dict=train_data)
#    a_rwd2 = sess.run(reward, feed_dict=train_data)
#    print("Epoch {:<12} Reward: {:<12.6f} ---> {:<12.6f}".format(epoch, a_rwd, a_rwd2))
    
    # === Tensorboard - start-2 === #
     # Run optimization / cost op (backprop / to get loss value) and summary nodes
    _, summary = sess.run([train_step,summary_op], feed_dict=train_data)
    # Write logs at every iteration
    writer.add_summary(summary,epoch)
    if epoch <= LOSSCHK_CNT:
        print("Run the command line:\n"
            "--> CD into your python install dir: {}\n"          
            "--> Execute: 'python -m tensorboard.main --logdir={} --host=127.0.0.1 --port=6006'\n"
            "Then open http://127.0.0.1:6006/ into your web browser".format(
                os.path.dirname(sys.executable)
                ,os.path.dirname(os.path.abspath( __file__ )) + os.sep + LOG_PATH)
            )
    # === Tensorboard - end-2 === #
    
    # == Save Model - start-1 == #
    if SAVE_MODELS:
        if epoch <= LOSSCHK_CNT:
            saver.save(sess, '{}'.format(SAVE_PATH + os.sep + 'model_out_' + TODAY), global_step=step)
        else:
            saver.save(sess, '{}'.format(SAVE_PATH + os.sep + 'model_out_' + TODAY), global_step=step, write_meta_graph=False)
    # == Save Model - end-1 == #

#---------------------------------------------------------------------------------------------------

plt.plot(dat_rwds)
plt.plot(imm_rwds)
plt.legend(['Discount Test Reward', 'Actual Test Reward'], loc=4)
plt.show()

if COMMISSION != 0:
    prev_weights = [[1 if idx == 0 else 0 for idx in range(N_OUT)]]
    stime = time.time()
    test_imm.at[:,PORT_W] = prev_weights * len(test_imm)
    b_x = np.reshape(test_imm[COLS_X], (-1,N_IN))
    b_y = np.reshape(test_imm[COLS_Y], (-1,N_OUT))
    for r in range(len(test_imm) - 1):
        feed_row = {X:  np.reshape(b_x.iloc[r,:], (-1,N_IN)),
                    Y_: np.reshape(b_y.iloc[r,:], (-1,N_OUT)),
                    PREV_W: np.reshape(prev_weights[-1], (-1, N_OUT))}
        weights, y_vec  = sess.run([Y, Y_], feed_dict=feed_row)
        y_vec_10 = (weights * 10 ** y_vec)
        w = y_vec_10 / np.sum(y_vec_10)
        prev_weights.append(w[0])
        b_x.at[r+1,PORT_W]      = w[0]
        test_dat.at[r+1,PORT_W] = w[0]
        test_imm.at[r+1,PORT_W] = w[0]
        #print(r / (time.time() - stime))
                           
    feed_imm = {X:  np.reshape(test_imm[COLS_X], (-1, N_IN)), 
                Y_: np.reshape(test_imm[COLS_Y], (-1, N_OUT)),
                PREV_W: np.reshape(prev_weights, (-1, N_OUT))}

    y1, y2, pw, f_rewards, f_loss = sess.run([Y, Y_, PREV_W, tensor_rwds, loss], 
                                         feed_dict = feed_imm)
else:
    
    y1, y2, f_rewards, f_loss = sess.run([Y, Y_, tensor_rwds, loss], 
                                         feed_dict = feed_imm)
y3 = y1 * y2
prof = [0]
for x in y3:
    prof.append(prof[-1]+sum(x))
    
plt.plot(prof)
plt.legend(['Actual Reward'], loc=4)
plt.show()

long_short = [[] for _ in range(3 if ALLOW_SHORTS else 2)]
props = [list(y1[:,i]) for i in range(len(y1[0]))]

for val in y1:
    val = list(val)
    long_short[0].append(val[0])
    if ALLOW_SHORTS:
        long_short[1].append(sum(val[1:N_COINS//2+1]))
        long_short[2].append(sum(val[N_COINS//2+1:]))
    else:
        long_short[1].append(sum(val[1:]))

rolling_window = 100
for i in range(len(long_short)):
    dat = pd.rolling_mean(pd.Series(long_short[i]),rolling_window)
    plt.plot(dat)
plt.legend(['USD', 'Long', 'Short'] if ALLOW_SHORTS else ['USD', 'Long'], loc=4)
plt.show()

for i in range(len(props)):
    dat = pd.rolling_mean(pd.Series(props[i]),rolling_window)
    plt.plot(dat)
plt.legend([x[x.index("_")+1:] for x in COLS_Y], loc=4)
plt.show()

result = pd.concat(
        
        [pd.DataFrame(y2, columns=COLS_Y),
         pd.DataFrame(y1, columns=["{}_%".format(x.replace("reward_","")) for x in COLS_Y])
         ]
        ,axis=1
        )

def showArray(arr, decimals = 3):
    return "[" + " ".join(["{1:.{0}f}".format(decimals, x) for x in arr]) + "]"

w = list(y1)

rewards     = []    # All Rewards     (Multiplicative)
log_rewards = []    # All Log Rewards (Additive)
prevW       = [1] + [0] * (N_OUT - 1) # Weights from previous period

STEP = 1

print("Iteration, PrevW, Action, PriceChange, NewW, Reward")
#------------------------------------------------------------------------------
for i in range(len(y2)):
    
    c = 0.0025
    
    for j in range(len(w[i])):
        w[i][j] = max(w[i][j],0)
        
    if i % STEP == 0:
        cw = [x for x in w[i]]
        
    rw     = 0    # Reward for this time step

    # Iterate through each asset and add each reward to the net reward
    #----------------------------------------------------------------
    for asset in range(len(cw)):

        # Transaction Cost
        tc       = (1 - c * abs((cw[asset] - prevW[asset])**1))
        if i % STEP != 0:
            tc = 1
        mult     = (10**y2[i][asset] - 1) + 1
            
        rw_asset = tc * (cw[asset]) * mult 
        rw      += rw_asset
    #----------------------------------------------------------------

    # Calculate what new weights will be after price move
    newW = [cw[A] * 10**y2[i][A] for A in range(len(cw))]
    newW = [x/sum(newW) for x in newW]
    
    #print(i, showArray(prevW), "-->", showArray(w[i]), "*", showArray(y[i]), "=", showArray(newW), " {{ {}{:.3f}% }}".format("+" if rw >= 1 else "", 100*rw-100))
    
    prevW = newW
    cw    = newW
    rewards.append(rw)
    log_rewards.append(math.log10(rw))
    
plt.plot(pd.Series(log_rewards).cumsum())
plt.plot(pd.Series(f_rewards).apply(lambda x : math.log10(math.exp(x))).cumsum())
plt.plot(pd.Series(test_imm.close_BTC / test_imm.close_BTC[0]).apply(lambda x : math.log10(x)))
#plt.plot(prof)
plt.show()

plt.plot(pd.Series(test_imm.close_BTC / test_imm.close_BTC[0]).apply(lambda x : math.log10(x)),
         pd.Series(log_rewards).cumsum(), 'ob')
plt.show()

if SAVE_MODELS:
    # This doesn't work atm
    pass'''answer = input("Do you want to save the Neural Network? (Y/N)")
    if "Y" in answer.upper():
        save_memory(sess, TRADING_PATH+"/NN.save")'''
