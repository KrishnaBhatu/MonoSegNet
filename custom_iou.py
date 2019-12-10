
import numpy as np
import cv2
import glob

input_images = []
for folder_name in glob.glob('C:/Users/hrish/Desktop/Testing/cityscapesScripts-master/MonoSegNet/sample_data/*'):
    for input_image_path in glob.glob(folder_name + "/*.png"):
        print(input_image_path)
        a = cv2.imread(input_image_path)
        print(a.shape)

eval_segm = cv2.imread('C:/Users/hrish/Desktop/Testing/cityscapesScripts-master/MonoSegNet/sample_data/augsburg/pred.png')
eval_segm = eval_segm/5
gt_segm = cv2.imread('C:/Users/hrish/Desktop/Testing/cityscapesScripts-master/MonoSegNet/sample_data/augsburg/gt.png')
#gt_segm = cv2.resize(gt_segm, (512 , 256), interpolation = cv2.INTER_NEAREST)
eval_segm = eval_segm[:,:,0]
gt_segm = gt_segm[:,:,0]

def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''
    check = check_size(eval_segm, gt_segm)
    if(check):
        cl, n_cl = extract_classes(gt_segm)
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

        sum_n_ii = 0
        sum_t_i  = 0

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :] # for coarse find the void class pixels multiple that and then carryon with the task
            curr_gt_mask = gt_mask[i, :, :]

            sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask)) #True Positive 
            sum_t_i  += np.sum(curr_gt_mask) # to find union it will be sum of both sets - intersection
    
        if (sum_t_i == 0):
            pixel_accuracy_ = 0
        else:
            pixel_accuracy_ = sum_n_ii / sum_t_i # penalize FN 

        return pixel_accuracy_
    return -1

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check = check_size(eval_segm, gt_segm)
    
    if check:
        cl, n_cl   = union_classes(eval_segm, gt_segm)
        _, n_cl_gt = extract_classes(gt_segm) # calc number of classes in gt
        eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl) # number of masks will be more for each eval and gt both

        IU = list([0]) * n_cl # making a list to store all the IOUs

        for i, c in enumerate(cl):
            curr_eval_mask = eval_mask[i, :, :]
            curr_gt_mask = gt_mask[i, :, :]
    
            if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
                continue

            n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask)) # intersection
            t_i  = np.sum(curr_gt_mask) # 
            n_ij = np.sum(curr_eval_mask) #union

            IU[i] = n_ii / (t_i + n_ij - n_ii) # preventing the double counting of intersection
    
        mean_IU_ = np.sum(IU) / n_cl_gt #classes in gt but not predicted by model are considered in acc but not vice versa
        return mean_IU_ # penalize FP and FN
    return -1

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl) # contains all the classes
    n_cl = len(cl) 

    return cl, n_cl


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)
    return cl, n_cl

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    print("eval = ",h_e,",",w_e)
    h_g, w_g = segm_size(gt_segm)
    print("gt = ",h_g,",",w_g)
    if (h_e != h_g) or (w_e != w_g):
        print("size mismatch")
        return False
    return True

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise
    return height, width

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl): # c is the class label number
        masks[i, :, :] = segm == c # here you will have 1s where the classes match

    return masks

def test():
    cl,n_cl = extract_classes(eval_segm)
    print("eval clases = {} and number = {}".format(cl,n_cl))
    cl,n_cl = extract_classes(gt_segm)
    print("Gt clases = {} and number = {}".format(cl,n_cl))
    check_size(eval_segm, gt_segm)
    print("Pixel Accuracy = ", pixel_accuracy(eval_segm, gt_segm))
    print("Mean IOU = ", mean_IU(eval_segm, gt_segm))
    
#test()