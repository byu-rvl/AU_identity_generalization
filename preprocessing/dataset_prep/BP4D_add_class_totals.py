# orignally from the ME-GraphAU repository, modified by Andrew Sumsion for the ELEGANT solution.

import numpy as np

# list_path_prefix = 'data/datasets/processed/BP4D/list/'
list_path_prefix = '/fslhome/andreww9/code/original_datasets/BP4D_croppped_MTCNN/list/'

'''
example of content in 'BP4D_train_label_fold1.txt':
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
'''

for fold in range(1,4):
    imgs_AUoccur = np.loadtxt(list_path_prefix + 'BP4D_train_label_fold'+str(fold)+'.txt')
    AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))
    for i in range(imgs_AUoccur.shape[1]):
        # AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0) / float(imgs_AUoccur.shape[0])
        AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0)

    np.savetxt(list_path_prefix+'BP4D_class_totals_fold'+str(fold)+'.txt', AUoccur_rate, fmt='%f', delimiter='\t')
