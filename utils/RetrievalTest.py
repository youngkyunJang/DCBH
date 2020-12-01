from utils.Dataload import *
from numpy.matlib import repmat

def cat_apcal(traingnd, testgnd, IX, top_N):
    [numtrain, numtest] = IX.shape

    apall = np.zeros(numtest)

    for i in range(numtest):
        y = IX[:, i]
        x = 0
        p = 0
        new_label = np.zeros((1,numtrain))
        new_label[traingnd.T == testgnd[i]] = 1
        num_retuen_NN = numtrain
        for j in range(num_retuen_NN):
            if new_label[0, y[j]] == 1:
                x = x + 1
                p = p + float(x) / (j + 1)
        if p == 0:
            apall[i] = 0
        else:
            apall[i] = p / x

    pall = np.zeros(numtest)
    for ii in range(numtest):
        y_1 = IX[:,ii]
        n = 0
        new_label_1 = np.zeros((1,numtrain))
        new_label_1[traingnd.T == testgnd[ii]] = 1
        for jj in range(top_N):
            if new_label_1[0,y_1[jj]] == 1:
                n = n + 1
        pall[ii] = 1.0 * n / top_N

    ap = np.mean(apall)
    p_topN = np.mean(pall)
    return ap, p_topN

def compactbit(b):
    b_mat = np.mat(b)
    [nSamples, nbits] = b_mat.shape
    nwords = int(np.ceil((float(nbits) / 8)))
    cb = np.zeros((nSamples, nwords), dtype=np.uint8)

    for i in range(nSamples):
        for j in range(nwords):
            temp = b[i, j * 8: (j + 1) * 8]
            value = convert(temp)
            cb[i, j] = value

    return cb

def convert(arr):
    arr_mat = np.mat(arr)
    [_, col] = arr_mat.shape
    value = 0
    for i in range(col):
        value = value + (2 ** i) * arr[i]

    return value

def evaluate_macro(Rel, Ret):
    Rel_mat = np.mat(Rel)
    numTest = Rel_mat.shape[1]

    precisions = np.zeros((numTest))
    recalls = np.zeros((numTest))

    retrieved_relevant_pairs = (Rel & Ret)

    for j in range(numTest):
        retrieved_relevant_num = len(retrieved_relevant_pairs[:, j][np.nonzero(retrieved_relevant_pairs[:, j])])
        # print 'retrieved_relevant_num=',retrieved_relevant_num
        retrieved_num = len(Ret[:, j][np.nonzero(Ret[:, j])])
        # print 'retrieved_num=',retrieved_num
        relevant_num = len(Rel[:, j][np.nonzero(Rel[:, j])])
        # print 'relevant_num=',relevant_num

        if retrieved_num:
            # print 1
            precisions[j] = float(retrieved_relevant_num) / retrieved_num

        else:
            precisions[j] = 0.0

        if relevant_num:
            recalls[j] = float(retrieved_relevant_num) / relevant_num

        else:
            recalls[j] = 0.0

    p = np.mean(precisions)
    r = np.mean(recalls)
    return p, r

def hammingDist(B1, B2):
    # look-up table:
    bit_in_char = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
                            3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
                            3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
                            2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
                            3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
                            5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
                            2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
                            4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
                            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
                            4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
                            5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
                            5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype=np.uint16)

    n1 = B1.shape[0]
    n2, nwords = B2.shape

    Dh = np.zeros((n1, n2), dtype=np.uint16)
    for i in range(n1):
        for j in range(nwords):
            y = (B1[i, j] ^ B2[:, j]).T
            Dh[i, :] = Dh[i, :] + bit_in_char[y]
    return Dh

def search(sess, x, training_flag, descriptors, train_x, train_y, test_x, test_y, TOP_K=50):
    pre_index = 0
    test_pre_index = 0
    iteration = 100
    test_iteration = 5
    train_data_num = np.shape(train_x)[0]
    test_data_num = np.shape(test_x)[0]
    batch_size = int(train_data_num/iteration)
    batch_size_test = int(test_data_num / test_iteration)

    for step in range(iteration + 1):
        if pre_index + batch_size < train_data_num:
            batch_x = train_x[pre_index: pre_index + batch_size]
        else:
            batch_x = train_x[pre_index:]
        pre_index += batch_size
        retrieval_feed_dict_train = {
            x: batch_x,
            training_flag: False
        }
        train_descriptors_batch = sess.run(descriptors, feed_dict=retrieval_feed_dict_train)
        if step == 0:
            train_descriptors = train_descriptors_batch
        else:
            train_descriptors = np.concatenate((train_descriptors, train_descriptors_batch), axis=0)

    for it in range(test_iteration + 1):
        if test_pre_index + batch_size_test < test_data_num:
            test_batch_x = test_x[test_pre_index: test_pre_index + batch_size_test]
        else:
            test_batch_x = test_x[test_pre_index:]
        test_pre_index += batch_size_test

        retrieval_feed_dict_test = {
            x: test_batch_x,
            training_flag: False
        }
        test_descriptors_batch = sess.run(descriptors, feed_dict=retrieval_feed_dict_test)
        if it == 0:
            test_descriptors = test_descriptors_batch
        else:
            test_descriptors = np.concatenate((test_descriptors, test_descriptors_batch), axis=0)

    print("Num Bits : %d"%(np.shape(train_descriptors[1])))
    gallery_binary_x = np.sign(train_descriptors)
    gallery_binary_x[gallery_binary_x == 0.0] = 1.0
    query_binary_x = np.sign(test_descriptors)
    query_binary_x[query_binary_x == 0.0] = 1.0

    train_binary_x, train_data_y = gallery_binary_x, train_y
    train_data_y.shape = (train_y.shape[0], 1)
    test_binary_x, test_data_y = query_binary_x, test_y
    test_data_y.shape = (test_y.shape[0], 1)

    train_y_rep = repmat(train_data_y, 1, test_data_y.shape[0])
    test_y_rep = repmat(test_data_y.T, train_data_y.shape[0], 1)
    cateTrainTest = (train_y_rep == test_y_rep)
    train_data_y = train_data_y + 1
    test_data_y = test_data_y + 1

    train_data_y = np.asarray(train_data_y, dtype=int)
    test_data_y = np.asarray(test_data_y, dtype=int)

    B = compactbit(train_binary_x)
    tB = compactbit(test_binary_x)

    hammTrainTest = hammingDist(tB, B).T

    hammRadius = 2
    Ret = (hammTrainTest <= hammRadius + 0.000001)
    [Pre, Rec] = evaluate_macro(cateTrainTest, Ret)
    print('Precision with Hamming radius_%d = %f'%(hammRadius, Pre))
    print('Recall with Hamming radius_%d = %f'%(hammRadius, Rec))

    HammingRank = np.argsort(hammTrainTest, axis=0)
    [MAP, p_topN] = cat_apcal(train_data_y, test_data_y, HammingRank, TOP_K)
    print('Precision of top %d returned = %f ' % (TOP_K, p_topN))
    return MAP
