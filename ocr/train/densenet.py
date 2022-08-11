from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Input, Flatten
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed


def conv_block(input, growth_rate, dropout_rate=None, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(growth_rate, (3,3), kernel_initializer='he_normal', padding='same')(x)
    if(dropout_rate):
        x = Dropout(dropout_rate)(x)
    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate=0.2, weight_decay=1e-4):
    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, droput_rate, weight_decay)
        x = concatenate([x, cb], axis=-1)
        nb_filter += growth_rate
    return x, nb_filter

def transition_block(input, nb_filter, dropout_rate=None, pooltype=1, weight_decay=1e-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)

    if(dropout_rate):
        x = Dropout(dropout_rate)(x)

    if(pooltype == 2):
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    elif(pooltype == 1):
        x = ZeroPadding2D(padding = (0, 1))(x)
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    elif(pooltype == 3):
        x = AveragePooling2D((2, 2), strides=(2, 1))(x)
    return x, nb_filter

def dense_cnn(input, nclass):

    _dropout_rate = 0.2 
    _weight_decay = 1e-4

    _nb_filter = 64
    # conv 64 5*5 s=2
    
    #卷积 输出通道64 卷积5*5 x，y方向步长2
    x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same',
                use_bias=False, kernel_regularizer=l2(_weight_decay))(input)
    
    # 分为8小层，每层又进行BN->Activation->Conv2D->Dropout，沿着通道拼接操作，
    # 通道数=nb_filter + growth_rate*8 = 64 + 8*8 = 128
    # 64 + 8 * 8 = 128
    print('1')
    print(x.shape)
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    print(x.shape)
    #BN->Activation->Conv2D->AveragePooling2D
    # 通道数128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    print(x.shape)
    # 分为8小层，每层又进行BN->Activation->Conv2D->Dropout，沿着通道拼接操作，
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    print(x.shape)
    #BN->Activation->Conv2D->AveragePooling2D
    # 192 -> 128
    x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay)
    print(x.shape)
    # 分为8小层，每层又进行BN->Activation->Conv2D->Dropout，沿着通道拼接操作，
    # 128 + 8 * 8 = 192
    x, _nb_filter = dense_block(x, 8, _nb_filter, 8, None, _weight_decay)
    print(x.shape)
    #
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    print(x.shape)
    x = Permute((2, 1, 3), name='permute')(x)
    print(x.shape)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    print(x.shape)
    y_pred = Dense(nclass, name='out', activation='softmax')(x)
    print(y_pred.shape)
    
    
    # basemodel = Model(inputs=input, outputs=y_pred)
    # basemodel.summary()
    
    #return y_pred, x
    """ 
    y_pred= densenet.dense_cnn(input, nclass)
    basemodel = Model(inputs=input, outputs=y_pred)

    modelPath = os.path.join(os.getcwd(), 'densenet/models/weights_densenet.h5')
    if os.path.exists(modelPath):
        basemodel.load_weights(modelPath) 
    
    input_pred = batch_complete
    
    input_gt = 
    
    _,x_pred = basemodel.predict(input)  
    
    _,x_gt = basemodel.predict(input_gt)
        
        
        
        """
    return y_pred

def dense_blstm(input):
    #😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊😊

    pass

input = Input(shape=(32, 280, 3), name='the_input')
dense_cnn(input, 5000)
