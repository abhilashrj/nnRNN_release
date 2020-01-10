import matplotlib.pyplot as plt
import pickle
import numpy as np

rnns = ['LSTM', 'RNN', 'NSRNN2','EXPRNN']
rnns_name = ['LSTM', 'RNN', 'nnRNN','EXPRNN']
# colors = ['lb', '']
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
for idx,rnn in enumerate(rnns):
    with open('./pickle_files/'+rnn+'.pkl', 'rb') as handle:
        losses = pickle.load(handle)
        plt.plot(np.arange(1,801), losses, label=str(rnns_name[idx]))
        plt.title("Copy task for T=200")
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.xticks([0,200,400,600,800])
#         plt.ylim([0,1])
#         plt.yticks([0,0.25,0.5,0.75,1])
ax.legend()
plt.show()
