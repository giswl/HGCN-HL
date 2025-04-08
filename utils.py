import scipy.io as sio
import numpy as np
import random
import torch
import torch.nn as nn
from operator import index, truediv
from sklearn.decomposition import PCA
from skimage.segmentation import slic,  felzenszwalb
from scipy.sparse import coo_matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, accuracy_score, classification_report, cohen_kappa_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_HSI_LiDAR_data(name):
    if name == 'MUUFL':
        X_HSI = sio.loadmat('data\MUUFL\hsi_data.mat')['hsi_data']
        X_LiDAR = sio.loadmat('data\MUUFL\lidar_data.mat')['lidar_data']
        Y = sio.loadmat('data\MUUFL\labels.mat')['labels']

        Y = np.where(Y==-1,0,Y)

        X_LiDAR = X_LiDAR[:,:,0]
        target_names = ['Trees','Grass_Pure','Grass_Groundsurface','Dirt_And_Sand', 'Road_Materials','Water',"Buildings'_Shadow",
                    'Buildings','Sidewalk','Yellow_Curb','ClothPanels']
        class_num = 11
        n_input_size = 64
    elif name == 'Houston2013':
        X_HSI = sio.loadmat('data\Houston2013\Houston_HS.mat')['Houston_HS']
        X_LiDAR = sio.loadmat('data\Houston2013\Houston_LiDAR.mat')['Houston_LiDAR']
        Y = sio.loadmat('data\Houston2013\Houston_Label.mat')['Houston_Label']
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass'
                        ,'Trees', 'Soil', 'Water', 
                        'Residential', 'Commercial', 'Road', 'Highway',
                        'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                        'Running Track']
        class_num = 15
        n_input_size = 144
    elif name == 'Trento':
        X_HSI = sio.loadmat('data\Trento\HSI.mat')['HSI']
        X_LiDAR = sio.loadmat('data\Trento\LiDAR.mat')['LiDAR']
        Y = sio.loadmat('data\Trento\Label.mat')['Label']
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass'
                        ,'Trees', 'Soil', 'Water', 
                        'Residential', 'Commercial', 'Road', 'Highway',
                        'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                        'Running Track']
        class_num = 6
        n_input_size = 63
    elif name == 'Augsburg':
        X_HSI = sio.loadmat(r'data\Augsburg\augsburg_hsi.mat')['augsburg_hsi']
        X_LiDAR = sio.loadmat(r'data\Augsburg\augsburg_sar.mat')['augsburg_sar']
        Y = sio.loadmat(r'data\Augsburg\augsburg_gt.mat')['augsburg_gt']
        X_LiDAR = X_LiDAR[:,:,3]
        target_names = ['Healthy grass', 'Stressed grass', 'Synthetic grass'
                        ,'Trees', 'Soil', 'Water', 
                        'Residential', 'Commercial', 'Road', 'Highway',
                        'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                        'Running Track']
        class_num = 7
        n_input_size = 180
    
    return X_HSI,X_LiDAR,Y,class_num,n_input_size,target_names
def get_dataset(FLAG):
    if FLAG == 1:
        name = 'MUUFL'
        (X_HSI,X_LiDAR,gt,class_num,n_input_size,target_names) = get_HSI_LiDAR_data(name)
        pass
    elif FLAG == 2:
        name = 'Houston2013'
        (X_HSI,X_LiDAR,gt,class_num,n_input_size,target_names) = get_HSI_LiDAR_data(name)
        pass
    elif FLAG == 3:
        name = 'Trento'
        (X_HSI,X_LiDAR,gt,class_num,n_input_size,target_names) = get_HSI_LiDAR_data(name)
        pass
    elif FLAG == 4:
        name = 'Augsburg'
        (X_HSI,X_LiDAR,gt,class_num,n_input_size,target_names) = get_HSI_LiDAR_data(name)
        pass
    return X_HSI,X_LiDAR, gt, class_num

def get_TrainValTest_Sets(seed: int, gt: np.array, class_count: int, train_ratio, 
                   val_ratio, samples_type: str = 'ratio', ):
    # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        train_number_per_class = []
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            
            rand_idx = random.sample(rand_list,
                                     np.ceil(samplesCount * train_ratio).astype('int32') + \
                                     np.ceil(samplesCount * val_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            train_number_per_class.append(np.ceil(samplesCount * train_ratio).astype('int32'))
            rand_real_idx_per_class = idx[rand_idx]
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx, dtype=object)
        train_data_index = []
        val_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = list(train_rand_idx[c])
            train_data_index = train_data_index + a[:train_number_per_class[c]]
            val_data_index = val_data_index + a[train_number_per_class[c]:]
            # for j in range(a.shape[0]):
            #     train_data_index.append(a[j][0:train_number_per_class])
            #     val_data_index.append()
        # train_data_index = np.array(train_data_index).reshape([-1])
        # val_data_index = np.array(val_data_index).reshape([-1])
        
        ##将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - val_data_index - background_idx
        
        # # 从测试集中随机选取部分样本作为验证集
        # val_data_count = np.ceil(val_ratio * (len(test_data_index) + len(train_data_index))).astype('int32')  # 验证集数量
        # val_data_index = random.sample(test_data_index, val_data_count)
        # val_data_index = set(val_data_index)
        
        # test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集
        
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    if samples_type == 'same_num':
        if int(train_ratio) == 0 or int(val_ratio) == 0:
            print("ERROR: The number of samples for train. or val. is equal to 0.")
            exit(-1)
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio)  # 每类相同数量样本,则训练比例为每类样本数量
            real_val_samples_per_class = int(val_ratio)  # 每类相同数量样本,则训练比例为每类样本数量
            
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class >= samplesCount:
                real_train_samples_per_class = samplesCount - 1
                real_val_samples_per_class = 1
            else:
                real_val_samples_per_class = real_val_samples_per_class if (
                                                                                       real_val_samples_per_class + real_train_samples_per_class) <= samplesCount else samplesCount - real_train_samples_per_class
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class + real_val_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
            if real_val_samples_per_class > 0:
                rand_real_idx_per_class_val = idx[rand_idx[-real_val_samples_per_class:]]
                val_rand_idx.append(rand_real_idx_per_class_val)
        
        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        
        val_data_index = []
        for c in range(val_rand_idx.shape[0]):
            a = val_rand_idx[c]
            for j in range(a.shape[0]):
                val_data_index.append(a[j])
        val_data_index = np.array(val_data_index)
        
        train_data_index = set(train_data_index)
        val_data_index = set(val_data_index)
        
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - val_data_index - background_idx
        
        # # 从测试集中随机选取部分样本作为验证集
        # val_data_count = int(val_samples)  # 验证集数量
        # val_data_index = random.sample(test_data_index, val_data_count)
        # val_data_index = set(val_data_index)
        
        # test_data_index = test_data_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    
    # # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    
    train_gt_map = train_samples_gt
    return train_data_index,test_data_index,val_data_index, train_gt_map

class CTrainValTest_Sets():
    train_data_index    = []
    test_data_index     = []
    val_data_index      = []
    train_gt_map        = []
    
    train_gt            = []
    val_gt              = []
    test_gt             = []

    def get_TrainValTest_Sets(self, cur_seed,
            gt,class_num,train_ratio,val_ratio, samples_type):
        (train_data_index,
         test_data_index, 
         val_data_index, 
         train_gt_map) = get_TrainValTest_Sets(cur_seed,
                gt,class_num, train_ratio,val_ratio, samples_type)
         
        gt_flatten = np.reshape(gt, -1)
        
        # label of training set
        train_gt = gt_flatten[train_data_index]
        train_gt = train_gt - 1
        train_gt = torch.from_numpy(train_gt).to(device)
        self.train_gt = train_gt.long()
    
        # label of valid set
        val_gt = gt_flatten[val_data_index]
        val_gt = val_gt - 1
        val_gt = torch.from_numpy(val_gt).to(device)
        self.val_gt = val_gt.long()
        
        # label of test set
        test_gt = gt_flatten[test_data_index]
        test_gt = test_gt - 1
        test_gt = torch.from_numpy(test_gt).to(device)
        self.test_gt = test_gt.long()
        
        self.train_data_index   = train_data_index
        self.test_data_index    =  test_data_index
        self.val_data_index     =  val_data_index
        self.train_gt_map       =  train_gt_map
        
        pass


class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.cost = nn.CrossEntropyLoss()
    def forward(self, X, Y):
        loss = self.cost(X, Y)
        return loss
    
class CResult():
    lossvalue = 0
    def get_permance(self, gt, output):
        gt = gt.detach().cpu().numpy()
        pre = np.argmax(output.detach().cpu().numpy(), axis=1)
        
        (oa, aa, kappa, 
            acc_perclass, confusion) = \
                get_HSI_performance(gt, pre)
        self.oa = oa
        self.aa = aa
        self.kappa = kappa
        self.acc_perclass = acc_perclass
        self.confusion = confusion
        # return self


def get_SLIC_Segs(Img, SLIC_scale):
    h,w = Img.shape[0],Img.shape[1]
    # SLIC Parameters
    ###### SLIC  #########
    n_SLIC_segs = h*w/SLIC_scale
    SLIC_segs = slic(Img, n_SLIC_segs)
    SLIC_segs = np.reshape(SLIC_segs,(-1))
    id_Segs = np.unique(SLIC_segs)
    n_Segs = len(id_Segs)
    H_SLIC = coo_matrix((h*w,n_Segs),dtype=np.int8).toarray()
    for j in range(len(id_Segs)):
        idx = np.where(SLIC_segs == id_Segs[j])
        H_SLIC[idx, j] = 1
    
    return H_SLIC
def get_felzenszwalb_Segs(Img, SLIC_scale):
    h,w = Img.shape[0],Img.shape[1]
    SLIC_segs = felzenszwalb(Img, scale=1,sigma=0.5,min_size=SLIC_scale)
    SLIC_segs = np.reshape(SLIC_segs,(-1))
    id_Segs = np.unique(SLIC_segs)
    n_Segs = len(id_Segs)
    H_SLIC = coo_matrix((h*w,n_Segs),dtype=np.int8).toarray()
    for j in range(len(id_Segs)):
        idx = np.where(SLIC_segs == id_Segs[j])
        H_SLIC[idx, j] = 1
    return H_SLIC
def obtain_H_from_HSI_with_LiDAR(HSI,LiDAR, scales):   
    h,w = HSI.shape[0],HSI.shape[1]
    Img = np.reshape(HSI,(-1,HSI.shape[2]))
    
    pca = PCA(n_components=3)  # 指定要保留的主成分数量
    Img = pca.fit_transform(Img)
    Img = np.reshape(Img, (h, w, 3))
    # plt.imshow(Img)
    X_3Band = np.reshape(Img,(HSI.shape[0],HSI.shape[1],3))
    min_value = np.min(X_3Band)
    max_value = np.max(X_3Band)
    X_3Band = (X_3Band - min_value) * 255 / (max_value - min_value)
    # 将数据转换为整型
    X_3Band = X_3Band.astype(np.uint8)
    H = None
    index = 0
    for scale in scales:
        H_HSI = get_SLIC_Segs(Img,scale)
        H_LiDAR = get_felzenszwalb_Segs(LiDAR,scale)
        H_HSI_LiDAR = np.concatenate([H_HSI,H_LiDAR],axis = 1)
        # H_HSI_LiDAR = H_LiDAR
        # H_HSI_LiDAR = H_HSI
        if index == 0:
            H = H_HSI_LiDAR
            index = 1
        else:
            H = np.concatenate([H,H_HSI_LiDAR],axis = 1)

    return H
def getnestedgeindex(x:torch.tensor,block_size:int = 500,k:int = 50,startedgeid = 100):
    edge_index = []
    edge_weight = []
    for i in range(0, x.size(0), block_size):
        block_features = x[i:i + block_size]
        # 计算当前块与所有特征的距离矩阵
        distance_matrix = torch.cdist(block_features, x)
        distance_matrix = torch.exp(-distance_matrix/0.1)
        # # 创建余弦相似度计算模块
        # cosine_similarity = nn.CosineSimilarity(dim=1)

        # # 计算余弦相似度
        # distance_matrix = cosine_similarity(block_features, x)
        # 获取前 k 个近邻索引
        top_values, top_indices = torch.topk(distance_matrix, k, largest=True)
        # 构建部分 edge_index
        partial_edge_index = torch.stack([top_indices.view(-1),torch.arange(startedgeid, startedgeid + block_features.size(0)).repeat_interleave(k).to(device) 
                                        ])
        # partial_edge_weight = torch.exp(-top_values/0.01)
        edge_index.append(partial_edge_index)
        edge_weight.append(top_values.view(-1))
        startedgeid = startedgeid+block_features.size(0)
    # 合并所有部分 edge_index
    edge_index = torch.cat(edge_index, dim=1)
    return edge_index,edge_weight
def getnestedgeindex(x:torch.tensor,block_size:int = 500,k:int = 10,startedgeid = 100,margin:float = 0.9):
    edge_index = []
    edge_weight = []
    for i in range(0, x.size(0), block_size):
        block_features = x[i:i + block_size]
        # 计算当前块与所有特征的距离矩阵
        distance_matrix = torch.cdist(block_features, x)
        distance_matrix = torch.exp(-distance_matrix/0.1)
        # # 创建余弦相似度计算模块
        # cosine_similarity = nn.CosineSimilarity(dim=1)

        # # 计算余弦相似度
        # distance_matrix = cosine_similarity(block_features, x)
        # 获取前 k 个近邻索引
        top_values, top_indices = torch.topk(distance_matrix, k, largest=True)
        # 构建部分 edge_index
        partial_edge_index = torch.stack([top_indices.view(-1),torch.arange(startedgeid, startedgeid + block_features.size(0)).repeat_interleave(k).to(device) 
                                        ])
        # partial_edge_weight = torch.exp(-top_values/0.01)
        edge_index.append(partial_edge_index)
        edge_weight.append(top_values.view(-1))
        startedgeid = startedgeid+block_features.size(0)
    # 合并所有部分 edge_index
    edge_index = torch.cat(edge_index, dim=1)
    return edge_index,edge_weight
def nparray_to_sparse_coo_tensor(A: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.sparse_coo_tensor:
    row, col = np.nonzero(A)
    data = A[row, col]
    indices = torch.from_numpy(np.vstack((row, col))).long()
    values = torch.from_numpy(data).float()
    return torch.sparse_coo_tensor(indices, values, torch.Size(A.shape)).to(device).coalesce()
def edge_weights(X,sigma:float=0.1,device:torch.device = torch.device("cpu")):
    """
    # edge weights: Gaussian kernel, row of S is vertex
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    R = torch.linalg.norm(X, axis=1)**2
    R = R.reshape(-1,1)
    X_XT = X @ X.T
    e = torch.ones((X.shape[0], 1),device=device)
    R = R @ e.T
    K = R+R.T-2*X_XT
    # A = torch.exp(-K/sigma)
    return K
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def get_HSI_performance(ytruth, ypredict):
    oa = accuracy_score(ytruth, ypredict)
    confusion = confusion_matrix(ytruth, ypredict)
    acc_perclass, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytruth, ypredict)
    return oa, aa, kappa, acc_perclass, confusion