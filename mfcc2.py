from smacpy import Smacpy
model = Smacpy("../Data/train", {'elephant1.wav':'elephant','elephant2.wav':'elephant','elephant3.wav':'elephant','train1.wav':'train', 'train2.wav':'train','train3.wav':'train','bg1_1.wav':'background','bg1_2.wav':'background','bg1_3.wav':'background'})
print model.classify('../Data/test/train_test.wav')
