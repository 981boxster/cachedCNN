def get_len_row(array):
    try:
        array.shape[1]
        len_row = array.shape[0]
    except IndexError:
        len_row = 1
    return len_row
    
def normalize_list_numpy(list_numpy):
    normalized_list = minmax_scale(list_numpy, feature_range=(0, 100))
    return normalized_list

def get_feat_gabor(tensor):
    feat_mean = np.zeros(len(tensor))
    feat_std = np.zeros(len(tensor))
    for i in range(len(tensor)):
        feat_mean[i] = np.mean(abs(tensor[i]))
        feat_std[i] = np.std(abs(tensor[i]))
    tensor_feat = np.vstack((feat_mean, feat_std)).ravel([-1])
    return tensor_feat

def cache_update(tensor, conv, result_class):
    global my_cache
    global my_cache_flag
    #print ("--------------------Cache is updating......-----------------------------------")
    #[0]class, [1]accumulate count, [2]mean distance, [3:]mean feature vector
    feat_vector = get_feat_gabor(tensor) # generate feat vector
    if(my_cache_flag[conv] == 0): # if cache table is empty
        new_row = np.array([[result_class, 1.0, -1.0]]) #class, count, distance
        new_row = np.insert(new_row, 3, feat_vector)
        new_row = np.array([new_row])
        my_cache[conv] = new_row
        my_cache_flag[conv] = 1
    elif(my_cache_flag[conv] == 1): #
        num_cached_class = get_len_row(my_cache[conv]) # number of rows (=number of cached classes)
        cached_classes = my_cache[conv][:,0:1]
        if (result_class in cached_classes): 
            index_class = np.where(cached_classes == result_class) 
            index_class = index_class[0][0]
            feat_new = feat_vector
            number_feat = my_cache[conv][index_class][1] # number of accumulated images
            dist_cached = my_cache[conv][index_class][2]
            feat_cached = my_cache[conv][index_class][3:] 
            dist_temp = distance.cosine(feat_new, feat_cached)
            if(dist_temp == 0.0):# 10):
                print ("This feature vector is already stored in cache. ")
            else:
                feat_mean = ((feat_cached * number_feat) + feat_new) / (number_feat + 1) # mean feature vector
                dist_new = distance.cosine(feat_mean, feat_new)
                
                if (dist_cached < 0): 
                    dist_mean = dist_new
                elif (dist_cached > 0): 
                    dist_mean = ((dist_cached * number_feat) + dist_new) / (number_feat + 1)
                my_cache[conv][index_class][1] = number_feat + 1
                my_cache[conv][index_class][2] = dist_mean 
                my_cache[conv][index_class][3:] = feat_mean
        else: 
            new_row = np.array([result_class, 1.0, -1.0]) # initial distance: -1
            new_row = np.insert(new_row, 3, feat_vector)
            my_cache[conv] = np.vstack((my_cache[conv], new_row)) 
    
    #print ("------------------------------------------------------------------------------")
    #print ("Current cache_",conv,"list below:\n", my_cache[conv][:,0:3])
        
def cache_search(tensor, conv): 
    global my_cache
    global my_cache_flag
    result = 0
    
    i = 0
    num_cached_classes = get_len_row(my_cache[conv]) # number of cached classes
    my_feat = get_feat_gabor(tensor) # input feature vector
    cached_dists = np.zeros(num_cached_classes) # cached distances
    my_dists = np.zeros(num_cached_classes) # distance between input feat vector and cached feat vector
    calced_diff_dists = np.zeros(num_cached_classes)

    for row in my_cache[conv]: # every caches (classes)
        cached_class = row[0]
        cached_dists[i] = row[2]
        cached_feat = row[3:] 
        my_dists[i] = distance.cosine(my_feat, cached_feat)
        i += 1

    if(my_dists.min() <= cached_dists[my_dists.argmin()]):
        result = int(my_cache[conv][my_dists.argmin()][0]) 
    else:
        result = 0
    return result#, diff
