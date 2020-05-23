import numpy as np
import pandas as pd

# AWA2_test_attributlabel: (6985, 85)
# AWA2_test_continuous_01_attributelabel: (6985, 85)
# AWA2_testlabel: (6985,)
# AWA2_train_continuous_01_attributelabel: (30337, 85)
# AWA2_trainlabel: (30337,)
# resnet101_testfeatures: (6985, 2048)
# resnet101_trainfeatures: (30337, 2048)
# AWA2_train_attributelabel: (30337, 85)


test = np.loadtxt("../new_split/food-label-vector-normed.txt")
test_classes_id_list = [0, 1, 3, 6, 8]
test_labels = []
for idx in test_classes_id_list:
   test_labels.append(test[idx])
test_labels = np.array(test_labels)
print(test_labels.shape)
print(test.shape)
np.save("../new_split/S_te_gt_food_Label_normed.npy", test_labels)
'''
test = np.load("../data/AWA2_test_continuous_01_attributelabel.npy")
print(test[0])
'''