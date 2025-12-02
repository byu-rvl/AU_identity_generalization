# orignally from the ME-GraphAU repository, modified by Andrew Sumsion for the ELEGANT solution.

import numpy as np

list_path_prefix = '/home/andreww9/fsl_groups/grp_RVL_AU/code/DISFA_LEFT_cropped_MTCNN/list/'

for fold in range(1,4):
    imgs_AUoccur = np.loadtxt(list_path_prefix + 'DISFA_train_label_fold'+str(fold)+'.txt')
    AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))
    for i in range(imgs_AUoccur.shape[1]):
        AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0) #/ float(imgs_AUoccur.shape[0])

    np.savetxt(list_path_prefix+'DISFA_class_totals_fold'+str(fold)+'.txt', AUoccur_rate, fmt='%f', delimiter='\t')
