# -*- coding: utf-8 -*-

from json import encoder
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, TimeDistributed, LSTM, Masking, concatenate, Dropout, Lambda, Bidirectional, Activation, BatchNormalization, Reshape, Add
from keras.constraints import unit_norm
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from datasets.SBU_data_batch import Dataloader
import numpy as np
from keras.engine import *
from tensorflow.python.ops import math_ops
import keras.backend as K
import tensorflow as tf
from attention_decoder_lstm_modified_state import AttentionDecoder
import copy



def bone_loss(y_true, y_predict):
    loss = 0.
    bones = [[13, 14], [14, 15], [3, 14], [4, 14], [9, 15], [10, 15],
                  [2, 3], [1, 2], [4, 5], [5, 6], [8, 9], [7, 8], [10, 11], [11, 12]]


    mask = K.any(K.not_equal(y_true, 0.), axis=-1)  # mask is bool type with shape (?,46)
    mask = K.cast(mask, dtype='float32')


    for bone_index in range(len(bones)):
        i = bones[bone_index][0]
        j = bones[bone_index][1]

        p_bone1 = y_predict[:, :, (3 * (i - 1)):(3 * i)]
        p_bone2 = y_predict[:, :, (3 * (j - 1)):(3 * j)]
        t_bone1 = y_true[:, :, (3 * (i - 1)):(3 * i)]
        t_bone2 = y_true[:, :, (3 * (j - 1)):(3 * j)]

        bone_length_input = K.sqrt(K.sum(K.square(p_bone2 - p_bone1), axis=-1) + 1e-10)
        bone_length_refs = K.sqrt(K.sum(K.square(t_bone2 - t_bone1), axis=-1) + 1e-10)

        loss_a = K.abs(bone_length_refs - bone_length_input) / (bone_length_refs + 1e-10)
        loss += tf.reduce_sum(loss_a, axis=-1)

    nframes = math_ops.count_nonzero(mask, axis=1)
    loss = loss / K.cast(nframes, dtype='float32')


    return K.mean(loss)

def contin_loss(y_true, y_predict, gap=2, alpha=0.1):

    mask = K.any(K.not_equal(y_true, 0.), axis=-1)  # mask is bool type with shape (?,46)
    mask = K.cast(mask, dtype='float32')
    mask = K.repeat(mask, y_predict.shape[-1])
    mask = tf.transpose(mask, [0, 2, 1])

    rerange_near = tf.concat([tf.expand_dims(y_predict[:, 0, :], 1), y_predict[:, :-1, :]], 1) * mask
    rerange_far = tf.concat([y_predict[:, :gap, :], y_predict[:, :-gap, :]], 1) * mask

    shape = y_predict.get_shape().as_list()
    shape[0] = -1
    shape[2] = shape[2] / 3
    shape.append(3)
    shape = [int(i) for i in shape]
    near_gap_4dim = tf.reshape(tf.squared_difference(y_predict * mask, rerange_near), shape)
    far_gap_4dim = tf.reshape(tf.squared_difference(y_predict * mask, rerange_far), shape)
    if gap == 2:
        far_gap_4dim = tf.concat([far_gap_4dim[:, 1:, :, :], tf.expand_dims(far_gap_4dim[:, 0, :, :], 1)], 1)
    else:
        far_gap_4dim = tf.concat([far_gap_4dim[:, (gap - 1):, :, :], far_gap_4dim[:, :(gap - 1), :, :]], 1)

    # first sum: for three coordinates a,b,c; second sum: for all joints
    loss1 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(near_gap_4dim, axis=-1) + 0.0000001), axis=-1)  # [173,93]
    loss2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(far_gap_4dim, axis=-1) + 0.0000001), axis=-1)  # [173,93]
    mask_alpha = mask[:, :, 0]
    mask_alpha = mask_alpha * tf.convert_to_tensor(alpha, dtype='float32')
    loss = loss1 - loss2
    loss = loss1
    loss += mask_alpha
    min_index = tf.minimum(math_ops.count_nonzero(loss1), math_ops.count_nonzero(loss2))
    loss *= math_ops.to_float(loss > 0)
    # sum: for all frames
    loss = tf.reduce_sum(loss[:, 1:(min_index + 1)], axis=-1)


    return K.mean(loss)

def L1_loss(y_true, y_predict):
    def selected_5_joints(y):
        y_5_joints = tf.concat([y[:, :, :3], y[:, :, 15:18], y[:, :, 18:21], y[:, :, 33:36], y[:, :, 42:45]], axis=-1)
        return y_5_joints
    print("HI")
    y_true_5joints = selected_5_joints(y_true)
    y_predict_5joints = selected_5_joints(y_predict)
    return tf.norm(y_predict - y_true, ord=1, keep_dims=False)

def L2_loss(y_true, y_predict):

    shape = y_predict.get_shape().as_list()
    shape[0] = -1
    shape[2] = shape[2] / 3
    shape.append(3)
    shape = [int(i) for i in shape]
    y_true = tf.reshape(y_true, shape)
    y_predict = tf.reshape(y_predict, shape)
    ans = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.squared_difference(y_true,y_predict),axis=-1)))# 1e-10

    return ans

def FD(y_true, y_predict): # frame distance
    ans = tf.reduce_sum(tf.squared_difference(y_true, y_predict), [1, 2])# 1e-10
    return ans


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class RemoveRandomness(Layer):

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(RemoveRandomness, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    def call(self, inputs, mask):
        if mask is not None:
            mask = K.repeat(mask, inputs.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
        return inputs * mask

    def compute_output_shape(self, input_shape):
        return input_shape


class InteractionGAN():
    def __init__(self):

        self.dropout = 0.5
        self.num_joints = 15
        self.vector_dim = 3 * self.num_joints
        self.hidden_dim = 200
        self.hidden_dim_slice = 40
        self.num_classes = 6
        self.gap = 5 # for far frames
        self.alpha = 0.1 # for gap between far frames (3-1) and near frames (2-1) of all joints###########
        self.data_loader = Dataloader(1)
        self.max_seqlength = self.data_loader.SBU_max_seqlength
        self.action_shape = (self.max_seqlength, self.vector_dim+self.num_classes)
        self.action_shape2 = (self.max_seqlength, self.vector_dim)
        self.action_shape_double = (self.max_seqlength, 2*self.vector_dim)

        # Build and compile the discriminator
        optimizer = RMSprop(lr=0.001)
        self.D = self.build_discriminator()
        self.CB = self.build_classifier_B()

        self.D.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0, 0.5],  # not using binary crossentropy, set weight to 0
            optimizer=optimizer,
            metrics=['accuracy'])
        

        self.CB.compile(
            loss=['categorical_crossentropy'],
            loss_weights=[1],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.G = self.build_generator()

        # The generator takes action A as input and generates counter action B
        actions_A = Input(shape=self.action_shape)
        actions_B  = self.G(actions_A)


        # For the combined model we will only train the generator
        self.D.trainable = True
        self.G.trainable = True
        self.CB.trainable = True
        # The discriminator takes generated images as input and determines validity
        [valid, label] = self.D(actions_B)

        # see output of inner layer
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=actions_A,
                              outputs=[valid, label, actions_B, actions_B, actions_B])

        self.combined.compile(loss=['binary_crossentropy', 'categorical_crossentropy', bone_loss, contin_loss, L1_loss],
                              optimizer=optimizer,
                              loss_weights=[1, 1, 0.01, 1, 1])

        self.combined.trainable = True

    def divide_skeleton_part(self, X):
        # two arms, two legs and one trunk, index from left to right, top to bottom
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] // 3, 3)
        assert (X.shape[2] == self.num_joints), ' skeleton must has %d joints' % self.num_joints
        sidx_list = [np.asarray([6, 5, 4]), np.asarray([7, 8, 9]),
                     np.asarray([12, 11, 10]), np.asarray([13, 14, 15]), np.asarray([1, 2, 3])]

        slic_idx = [it * X.shape[3] for it in [0, 3, 3, 3, 3, 3]]
        slic_idx = np.cumsum(slic_idx)

        X_new = np.zeros((X.shape[0], X.shape[1], slic_idx[-1]))
        for idx, sidx in enumerate(sidx_list):
            sidx = sidx - 1  # index starts from 0
            X_temp = X[:, :, sidx, :]
            X_new[:, :, slic_idx[idx]:slic_idx[idx + 1]] = np.reshape(X_temp, (
            X_temp.shape[0], X_temp.shape[1], X_temp.shape[2] * X_temp.shape[3]))
        return X_new

    def restore_skeleton_part(self, X):
        # two arms, two legs and one trunk, index from left to right, top to bottom
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] // 3, 3)
        assert (X.shape[2] == self.num_joints), ' skeleton must has %d joints' % self.num_joints
        sidx_list = [np.asarray([13, 14, 15]), np.asarray([3, 2, 1]),
                     np.asarray([4, 5, 6]), np.asarray([9, 8, 7]), np.asarray([10, 11, 12])]

        slic_idx = [it * X.shape[3] for it in [0, 3, 3, 3, 3, 3]]
        slic_idx = np.cumsum(slic_idx)

        X_new = np.zeros((X.shape[0], X.shape[1], slic_idx[-1]))
        for idx, sidx in enumerate(sidx_list):
            sidx = sidx - 1  # index starts from 0
            X_temp = X[:, :, sidx, :]
            X_new[:, :, slic_idx[idx]:slic_idx[idx + 1]] = np.reshape(X_temp, (
            X_temp.shape[0], X_temp.shape[1], X_temp.shape[2] * X_temp.shape[3]))
        return X_new


    def build_generator(self):
        encoder_inputs_A = Input(shape=self.action_shape)
        yinp = encoder_inputs_A[:,:,45:]
        new_slic_idx = [it*3 for it in [0, 3, 3, 3, 3, 3,2]]

        new_slic_idx = np.cumsum(new_slic_idx)
        hierLSTM = []
        hierLSTM_out = []
        hierState_h = []
        hierState_c = []
        dummyLSTM = []

        def slice(x, index1, index2):
            return x[:, :, index1:index2]
        for slc_id in range(0, len(new_slic_idx) - 1):

            sliced_data = Lambda(slice, arguments={'index1': new_slic_idx[slc_id], 'index2':new_slic_idx[slc_id+1]})(encoder_inputs_A)
            sliced_data = Masking(mask_value=0.)(sliced_data)
            shape = (sliced_data.shape)[2]
            shape = int(shape)
            sliced_data = Dense(shape+50)(sliced_data)
            hold = Bidirectional(LSTM(self.hidden_dim_slice, return_sequences=True, return_state=True))(sliced_data)
            slice_LSTM = hold[0]
            state_h = concatenate([hold[1],hold[3]],axis=-1)
            state_c = concatenate([hold[2],hold[4]],axis=-1)
            hierLSTM.append(slice_LSTM)
            hierState_h.append(state_h)
            hierState_c.append(state_c)

        def generate_y0(x):
            y0 = K.zeros_like(x)  # (samples, timesteps, input_dims)
            y0 = K.sum(y0, axis=(1, 2))  # (samples, )
            y0 = K.expand_dims(y0)  # (samples, 1)
            y0 = K.tile(y0, [1, 240*2])#1024])
            return y0

        hierLSTM_out = concatenate(hierLSTM, axis=-1)
        self.store = hierLSTM_out
        state_y = Lambda(generate_y0)(hierLSTM_out)
        hierState_h_out = concatenate(hierState_h, axis=-1)
        hierState_c_out = concatenate(hierState_c, axis=-1)
        decoder_initial = [state_y, hierState_h_out, hierState_c_out]
        aLSTM_out = AttentionDecoder(2*5*self.hidden_dim_slice+(40*2), 2*self.hidden_dim+(40*2), return_sequences=True)(hierLSTM_out, initial_state=decoder_initial)
        inter_outputs1 = Dense(self.vector_dim)(aLSTM_out)
        inter_outputs2 = RemoveRandomness()(inter_outputs1)
        decoder_outputs = NonMasking()(inter_outputs2)
        print(decoder_outputs.shape)
        return Model(encoder_inputs_A, [decoder_outputs])

    def build_discriminator(self):

        action_B = Input(shape=self.action_shape2)   
        self.modelD = Sequential()
        self.modelD.add(NonMasking(input_shape=self.action_shape2))
        self.modelD.add(Masking(mask_value=0., input_shape=self.action_shape2))
        self.modelD.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=True, kernel_constraint=unit_norm())))
        self.modelD.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=True, kernel_constraint=unit_norm())))
        self.modelD.add(NonMasking())
        self.modelD.add(Flatten())

        self.modelD.summary()

        features = self.modelD(action_B)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax", kernel_constraint=unit_norm())(features)
        print("HELLOD")
        return Model(action_B, [valid, label])
    

    def build_classifier_B(self):
        action_B = Input(shape=self.action_shape2)   
        self.modelDdd = Sequential()
        self.modelDdd.add(NonMasking(input_shape=self.action_shape2))
        self.modelDdd.add(Masking(mask_value=0., input_shape=self.action_shape2))
        self.modelDdd.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=True, kernel_constraint=unit_norm())))
        self.modelDdd.add(Bidirectional(LSTM(self.hidden_dim, return_sequences=True, kernel_constraint=unit_norm())))
        #self.modelD.add(Dropout(0.5))
        #self.modelD.add(BatchNormalization(momentum=0.8))
        self.modelDdd.add(NonMasking())
        self.modelDdd.add(Flatten())

        self.modelDdd.summary()

        features = self.modelDdd(action_B)
        # valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes , activation="softmax", kernel_constraint=unit_norm())(features)########################
        print("HELLOD")
        return Model(action_B, [label])
    
    def train(self, epochs, batch_size):

        self.test_fd = []
        
        half_batch = 1 
        cw1 = {0.9: 1, 0.1: 1} #cw1 = {0.1: 1, 0.9: 1}
        cw2 = {i: 0.375 for i in range(self.num_classes)}
        cw2[self.num_classes] = 1.0 / half_batch
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))*0.9 #####
        fake = np.zeros((batch_size, 1))+0.1 #####

        AFD = np.zeros(self.num_classes)
        AFD_max = np.full(self.num_classes, np.inf)
        bestg = [1000 for i in range(self.num_classes)]
        for epoch in range(epochs):

            for batch_i, (A_trains, B_trains, y_train, filename_train) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Translate actions to counter actions
                A_trains = self.divide_skeleton_part(A_trains)
                B_trains = self.divide_skeleton_part(B_trains)

                ycat = to_categorical(y_train, num_classes=self.num_classes)
                ydtrain = np.array(y_train)

                A_traind = A_trains.tolist()
                ycatd = ycat.tolist()

                for i in range(len(A_trains)):
                    ycatdz = to_categorical(ydtrain[i][0], num_classes=self.num_classes)
                    for j in range(len(A_trains[i])):
                        for x in ycatdz:
                            A_traind[i][j].append(x)

                A_traind = np.array(A_traind)
                fake_B = self.G.predict(A_traind)
                Acc_loss = self.CB.train_on_batch(fake_B,[ycat],class_weight=[1])

                # One-hot encoding of labels
                real_labels = to_categorical(y_train, num_classes=self.num_classes + 1)
                fake_labels = np.concatenate(
                    (np.full((batch_size, self.num_classes), 0.1 / self.num_classes), np.full((batch_size, 1), 0.9)),
                    axis=-1)

                # Train the discriminator
                cw2={i: batch_size/y_train.count([i]) if y_train.count([i])!=0 else batch_size for i in range(self.num_classes)}
                cw2[self.num_classes]=2 
                D_loss_fake = self.D.train_on_batch(fake_B[:,:,:46], [fake, fake_labels], class_weight=[0, cw2]) # class weight for binary discriminator is 0
                D_loss_real = self.D.train_on_batch(B_trains, [valid, real_labels], class_weight=[0, cw2])
                D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
                
                # --------------------- 
                #  Train Generator
                # ---------------------

                G_loss = self.combined.train_on_batch(A_traind, [valid, real_labels, B_trains, B_trains, B_trains], class_weight=[cw1, cw2, None, None, None])
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch+1, epochs, batch_i+1, self.data_loader.n_batches, D_loss[2], G_loss[0]))

            #-----------------
            #  Test Dataloader
            #-----------------
            A_tests, B_tests, y_tests, filename_test, num_frames_test, num_seqs_test = self.data_loader.load_test()
            A_tests = self.divide_skeleton_part(A_tests)
            B_tests = self.divide_skeleton_part(B_tests)

            ydtest = np.array(y_tests)
            A_testd = A_tests.tolist()
            #append labels
            for i in range(len(A_testd)):
                ycatdz = to_categorical(ydtest[i][0], num_classes=self.num_classes)
                for j in range(len(A_testd[i])):
                    for x in ycatdz:
                        A_testd[i][j].append(x)

            A_testd = np.array(A_testd)
            fake_B_tests = self.G.predict(A_testd)
            
            ## AFD
            assert np.sum(num_seqs_test) == len(B_tests)
            with tf.Session() as sess:
                fd = sess.run(FD(B_tests, fake_B_tests))
            frameDis = fd/num_frames_test
            for i in range(self.num_classes):
                AFD[i] = np.sum(frameDis[np.squeeze(np.array(y_tests)) == i])
                AFD[i] /= num_seqs_test[i]
            
            print("Kicking AFD: {0:.4f}; Pushing AFD: {1:.4f}; Shaking Hands AFD: {2:.4f}; Hugging AFD: {3:.4f}; Exchanging Object AFD: {4:.4f}; Punching AFD: {5:.4f}".format(
                    AFD[0], AFD[1], AFD[2], AFD[3], AFD[4], AFD[5]))

            
            #code snippet to save synthesized reaction according to their corresponding label in test set
            tosaveB = fake_B_tests
            realB = B_tests
            realA = A_tests
            tosaveB = self.restore_skeleton_part(tosaveB)
            realB = self.restore_skeleton_part(realB)
            realA = self.restore_skeleton_part(realA)
            np.save("wholethingFBcur",tosaveB)
            np.save("wholethingRBcur",realB)
            np.save("wholethingRAcur",realA)
            np.save("wholethingycur",y_tests)

            A_testd = A_tests.tolist()
            for i in range(len(A_testd)):
                    # ycatdz = to_categorical(2, num_classes=self.num_classes)
                    ycatdz = [0,0,1,1,0,0] # Multi-Hot Vector of shaking hands and hugging
                    for j in range(len(A_testd[i])):
                        for x in ycatdz:
                            if(x==1):
                                x=1
                            A_testd[i][j].append(x)

            A_testd = np.array(A_testd)
            fake_B_tests = self.G.predict(A_testd)
            tosaveB = fake_B_tests
            realB = B_tests
            realA = A_tests
            tosaveB = self.restore_skeleton_part(tosaveB)
            realB = self.restore_skeleton_part(realB)
            realA = self.restore_skeleton_part(realA)
            np.save("wholethingFB23",tosaveB)
            np.save("wholethingRB23",realB)
            np.save("wholethingRA23",realA)
            np.save("wholethingy23",y_tests)

            A_testd = A_tests.tolist()
            for i in range(len(A_testd)):
                    ycatdz = to_categorical(3, num_classes=self.num_classes) #Multi-hot vector of hugging synthesized reaction
                    for j in range(len(A_testd[i])):
                        for x in ycatdz:
                            if(x==1):
                                x=1
                            A_testd[i][j].append(x)
            A_testd = np.array(A_testd)
            fake_B_tests = self.G.predict(A_testd)
            tosaveB = fake_B_tests
            realB = B_tests
            realA = A_tests
            tosaveB = self.restore_skeleton_part(tosaveB)
            realB = self.restore_skeleton_part(realB)
            realA = self.restore_skeleton_part(realA)
            np.save("wholethingFBguid3",tosaveB)
            np.save("wholethingRBguid3",realB)
            np.save("wholethingRAguid3",realA)
            np.save("wholethingyguid3",y_tests)


            if np.mean(AFD_max)>np.mean(AFD):
                AFD_max=copy.deepcopy(AFD)
                epoch_best=epoch+1 
            #Might need to rerun training couple of times to get desired AFD.
            A_testd = A_tests.tolist()
            for i in range(len(A_testd)):
                    ycatdz = to_categorical(3, num_classes=self.num_classes) #Multi-hot vector of hugging synthesized reaction
                    for j in range(len(A_testd[i])):
                        for x in ycatdz:
                            if(x==1):
                                x=1
                            A_testd[i][j].append(x)
            A_testd = np.array(A_testd)
            fake_B_tests = self.G.predict(A_testd)
            tosaveB = fake_B_tests
            realB = B_tests
            realA = A_tests
            tosaveB = self.restore_skeleton_part(tosaveB)
            realB = self.restore_skeleton_part(realB)
            realA = self.restore_skeleton_part(realA)
            np.save("wholethingFBguid3b",tosaveB)
            np.save("wholethingRBguid3b",realB)
            np.save("wholethingRAguid3b",realA)
            np.save("wholethingyguid3b",y_tests)
            
            print("Best results achieve at epoch%d:" % int(epoch_best))
            print("Kicking AFD: {0:.4f}; Pushing AFD: {1:.4f}; Shaking Hands AFD: {2:.4f}; Hugging AFD: {3:.4f}; Exchanging Object AFD: {4:.4f}; Punching AFD: {5:.4f}".format(AFD_max[0], AFD_max[1], AFD_max[2], AFD_max[3], AFD_max[4], AFD_max[5]))

if __name__ == '__main__':

    gan = InteractionGAN()
    gan.train(epochs=2000, batch_size=16) 
   


