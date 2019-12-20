
import numpy as np
import cv2
import glob

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

def find_IU(eval_segm, gt_segm,IU,IU_occurences):

    check = check_size(eval_segm, gt_segm)
    
    assert check == True

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm) # calc number of classes in gt
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl) # number of masks will be more for each eval and gt both

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask)) # intersection
        t_i  = np.sum(curr_gt_mask) #
        n_ij = np.sum(curr_eval_mask) #union
        value = n_ii/(t_i+n_ij-n_ii)
        if(value > 1):
            print(n_ii," ",t_i," ", n_ij)
        IU[c] += value # preventing the double counting of intersection
        #if(IU[c] == 0):
        #    print(n_ii," ",t_i," ", n_ij)
        IU_occurences[c] += 1
    #mean_IU_ = np.sum(IU) / n_cl_gt #classes in gt but not predicted by model are considered in acc but not vice versa
    return (IU, IU_occurences) # penalize FP and FN

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
    #print("eval = ",h_e,",",w_e)
    h_g, w_g = segm_size(gt_segm)
    #print("gt = ",h_g,",",w_g)
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
    #cl,n_cl = extract_classes(eval_segm)
    #print("eval clases = {} and number = {}".format(cl,n_cl))
    #cl,n_cl = extract_classes(gt_segm)
    #print("Gt clases = {} and number = {}".format(cl,n_cl))
    #check_size(eval_segm, gt_segm)
    #print("Pixel Accuracy = ", pixel_accuracy(eval_segm, gt_segm))
    IU = np.zeros(35) # making a list to store all the IOUs
    IU_occurences = np.zeros(35) # for the 35 clasees    
    num = 0
    with open('/home/adi_leo96_av/MonoSegNet/test_index.txt','r') as f:
        for line in f:
            paths = line.split(' ')
            paths[1] = paths[1][:-1]
            #print(paths)
            eval_segm = cv2.imread(paths[0])
            gt_segm = cv2.imread(paths[1])
            if(eval_segm is not None and gt_segm is not None):
                gt_segm = cv2.resize(gt_segm,(512,256),interpolation = cv2.INTER_NEAREST)
                (IU,IU_occurences) = find_IU(eval_segm[:,:,0], gt_segm[:,:,0],IU,IU_occurences)
                num += 1
                print(num)
                #if(num == 10):
                #    break
            else:
                print("one of the image is empty")
    sum = 0
    # for i in range(34):
    #     IU[i] = IU[i]/IU_occurences[i] 
    #    sum += IU[i]
    print("IU_before change", IU)
    print("IU_occurences before change", IU_occurences)
    void_list = [0,1,2,3,4,5,6]
    for ind in void_list:
        IU_occurences[ind] = 0
    IU_occurences_ = IU_occurences[np.nonzero(IU_occurences)]
    IU = IU[np.nonzero(IU_occurences)] #Takes care of the situation where in the intersection area is zero
    IOU = IU/IU_occurences_
    print("IU = ", IU)
    print("IU_occurences",IU_occurences_)
    print("IOU = ", IOU)
    print("mIOU = ", np.mean(IOU))
test()
