
from operator import delitem
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences


class Dataloader():
    def __init__(self, num_fold):
        self.SBU_bones = [[1, 2], [2, 3], [2, 4], [2, 7], [3, 10], [3, 13], [4, 5], [5, 6], [7, 8], [8, 9], [10, 11], [11, 12],
                 [13, 14], [14, 15]]

        self.SBU_bone_length_ref = [0.11181145, 0.11415315, 0.10849458, 0.10848793, 0.13271541, 0.13271541,
                           0.16009105, 0.14876385, 0.16009105, 0.14876385, 0.22243133, 0.21444634,
                           0.22243133, 0.21444634]
        self.SBU_max_seqlength = 46
        self.path = "./datasets/SBU/normalized_7_fold/11fold/"
        self.SBU_labels = 6

    def load_batch(self, batch_size=1):
        files_A = os.listdir(self.path + "train_A")
        self.n_batches = int(len(files_A) / batch_size)
        total_samples = self.n_batches * batch_size
        files_A = np.random.choice(files_A, total_samples, replace=False) # reorder the sequences so that the model will see all the training samples
        for i in range(self.n_batches):
            batch = files_A[i * batch_size:(i + 1) * batch_size]
            A_train, B_train, y_train, filename_train = [], [], [], []  
            for j in batch:
                #synthetic dataset loader with their class labels
                t = j[12:]
                if(j[0]=="j"):
                    if(t=="0"):
                            y_train.append([int(5)])
                    elif(t=="1"):
                            y_train.append([int(1)])
                    elif(t=="2"):
                            y_train.append([int(2)])
                    elif(t=="3"):
                            y_train.append([int(0)])
                    elif(t=="4"):
                            y_train.append([int(5)])
                    elif(t=="5"):
                            y_train.append([int(4)])
                    elif(t=="6"):
                            y_train.append([int(3)])
                    elif(t=="7"):
                            y_train.append([int(0)])
                    elif(t=="8"):
                            y_train.append([int(5)])
                    elif(t=="9"):
                            y_train.append([int(5)])
                    elif(t=="10"):
                            y_train.append([int(4)])
                    elif(t=="11"):
                            y_train.append([int(4)])
                    elif(t=="12"):
                            y_train.append([int(2)])
                    elif(t=="13"):
                            y_train.append([int(2)])
                    elif(t=="14"):
                            y_train.append([int(5)])
                    elif(t=="15"):
                            y_train.append([int(0)])
                    elif(t=="16"):
                            y_train.append([int(1)])
                    elif(t=="17"):
                            y_train.append([int(5)])
                    elif(t=="18"):
                            y_train.append([int(0)])
                    elif(t=="19"):
                            y_train.append([int(4)])
                    elif(t=="20"):
                            y_train.append([int(3)])
                    elif(t=="21"):
                            y_train.append([int(5)])
                    elif(t=="22"):
                            y_train.append([int(1)])
                    elif(t=="23"):
                            y_train.append([int(1)])
                    elif(t=="24"):
                            y_train.append([int(2)])
                    elif(t=="25"):
                            y_train.append([int(0)])
                    elif(t=="26"):
                            y_train.append([int(0)])
                    elif(t=="27"):
                            y_train.append([int(5)])
                    elif(t=="28"):
                            y_train.append([int(1)])
                    elif(t=="29"):
                            y_train.append([int(0)])
                    elif(t=="30"):
                            y_train.append([int(4)])
                    elif(t=="31"):
                            y_train.append([int(4)])
                    elif(t=="32"):
                            y_train.append([int(5)])
                    elif(t=="33"):
                            y_train.append([int(4)])
                    elif(t=="34"):
                            y_train.append([int(0)])
                    elif(t=="35"):
                            y_train.append([int(3)])
                    elif(t=="36"):
                            y_train.append([int(2)])
                    elif(t=="37"):
                            y_train.append([int(5)])
                    elif(t=="38"):
                            y_train.append([int(1)])
                    elif(t=="39"):
                            y_train.append([int(0)])
                    elif(t=="40"):
                            y_train.append([int(4)])
                    elif(t=="41"):
                            y_train.append([int(0)])
                    elif(t=="42"):
                            y_train.append([int(1)])
                    elif(t=="43"):
                            y_train.append([int(3)])
                    elif(t=="44"):
                            y_train.append([int(4)])
                    elif(t=="45"):
                            y_train.append([int(4)])
                    elif(t=="46"):
                            y_train.append([int(1)])
                    elif(t=="47"):
                            y_train.append([int(0)])
                    elif(t=="48"):
                            y_train.append([int(3)])
                    elif(t=="49"):
                            y_train.append([int(1)])
                    elif(t=="49"):
                            y_train.append([int(2)])

                    
                    try:
                        A_train.append(np.loadtxt(self.path + "train_A/" + j))
                    except:
                        A_train.append(np.loadtxt(self.path + "train_A/" + j,delimiter=","))
                    B_train.append(np.loadtxt(self.path + "train_B/" + "jointB"+ j[6:],delimiter=","))
                    filename_train.append(j[6:])
                else:
                #preprocessed SBU dataloader
                    A_train.append(np.loadtxt(self.path + "train_A/" + j))
                    B_train.append(np.loadtxt(self.path + "train_B/" + j[:14] + "_B.txt"))
                    y_train.append([int(j[2])-3])          #################################################
                    filename_train.append(j[:12])
                
            A_trains = []
            # pad A_trains with 100 of the same #frames (SBU:46)
            for k in range(len(A_train)):
                A_trains.append(np.transpose(pad_sequences(np.transpose(A_train[k]),
                                                           maxlen=self.SBU_max_seqlength,
                                                           dtype='float32',
                                                           padding='post', value=0.)))
            A_trains = np.array(A_trains)

            B_trains = []
            # pad B_trains with 0 of the same #frames (SBU:46)
            for h in range(len(B_train)):
                B_trains.append(np.transpose(pad_sequences(np.transpose(B_train[h]),
                                                           maxlen=self.SBU_max_seqlength,
                                                           dtype='float32',
                                                           padding='post', value=0.)))
            B_trains = np.array(B_trains)



            yield A_trains, B_trains, y_train, filename_train


    def load_test(self):
        self.count1 =0
        self.count2 = 0
        files_test_A = os.listdir(self.path + "test_A")
        A_test, B_test, y_test, filename_test, num_frames_test = [], [], [], [], []
        num_seqs_test = np.zeros(self.SBU_labels, dtype=int)
        self.count3 = len(files_test_A)
        for i in files_test_A:
            action = -1
            t = i[12:]
            #synthetic dataset loader with their class labels
            if(i[0]=="j"):
                self.count1+=1
                if(t=="0"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="1"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="2"):
                        y_test.append([int(2)])
                        action = 2
                elif(t=="3"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="4"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="5"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="6"):
                        y_test.append([int(3)])
                        action = 3
                elif(t=="7"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="8"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="9"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="10"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="11"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="12"):
                        y_test.append([int(2)])
                        action = 2
                elif(t=="13"):
                        y_test.append([int(2)])
                        action = 2
                elif(t=="14"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="15"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="16"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="17"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="18"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="19"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="20"):
                        y_test.append([int(3)])
                        action = 3
                elif(t=="21"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="22"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="23"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="24"):
                        y_test.append([int(2)])
                        action = 2
                elif(t=="25"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="26"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="27"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="28"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="29"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="30"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="31"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="32"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="33"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="34"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="35"):
                        y_test.append([int(3)])
                        action = 3
                elif(t=="36"):
                        y_test.append([int(2)])
                        action = 2
                elif(t=="37"):
                        y_test.append([int(5)])
                        action = 5
                elif(t=="38"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="39"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="40"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="41"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="42"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="43"):
                        y_test.append([int(3)])
                        action = 3
                elif(t=="44"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="45"):
                        y_test.append([int(4)])
                        action = 4
                elif(t=="46"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="47"):
                        y_test.append([int(0)])
                        action = 0
                elif(t=="48"):
                        y_test.append([int(3)])
                        action = 3
                elif(t=="49"):
                        y_test.append([int(1)])
                        action = 1
                elif(t=="49"):
                        y_test.append([int(2)])
                        action = 2
                try:
                        A_test.append(np.loadtxt(self.path + "test_A/" + i))
                except:
                        A_test.append(np.loadtxt(self.path + "test_A/" + i,delimiter=","))
                B_test.append(np.loadtxt(self.path + "test_B/" + "jointB"+ i[6:],delimiter=","))
                filename_test.append(i[6:])
            else:
        #preprocessed SBU dataloader
                self.count2+=1
                A_test.append(np.loadtxt(self.path + "test_A/" + i))
                B_test.append(np.loadtxt(self.path + "test_B/" + i[:14] + "_B.txt"))
                y_test.append([int(i[2])-3])
                filename_test.append(i[:14])
            # count how many frames in test seq to compute AFD
            temp = np.loadtxt(self.path + "test_A/" + i)
            num_frames_test.append(temp.shape[0])
            # count how many seqs are within the same label to compute AFD for each label   
            num_seqs_test[int(i[2])-3] += 1


        A_tests, B_tests = [], []
        # pad A_tests with 100 of the same #frames (SBU:46)
        for k in range(len(A_test)):
            A_tests.append(np.transpose(pad_sequences(np.transpose(A_test[k]),
                                                      maxlen=self.SBU_max_seqlength,
                                                      dtype='float32',
                                                      padding='post', value=0.)))

        A_tests = np.array(A_tests)

        for h in range(len(B_test)):
            B_tests.append(np.transpose(pad_sequences(np.transpose(B_test[h]),
                                                      maxlen=self.SBU_max_seqlength,
                                                      dtype='float32',
                                                      padding='post', value=0.)))

        B_tests = np.array(B_tests)



        return A_tests, B_tests, y_test, filename_test, num_frames_test, num_seqs_test

# x = Dataloader(1)

# a = x.load_batch(6)
# for batch_i, (A_trains, B_trains, y_train, filename_train) in enumerate(a):
#     print(batch_i)  
#     # print(A_trains, B_trains, y_train, filename_train[0])
