# coding: utf-8
# it is the first attempt in zero-shot-learning in tianchi match . attempt to use gcn with
# a dense graph constructed by the Euclidean distance
# it is a baseline and I will try to optimize the algorithm or have a more creative one.
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
#from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential
from gcn import *
import os
from sklearn.model_selection import train_test_split


TRAIN_PATH_ROOT = '../DatasetA_train_20180813/'
TEST_PATH_ROOT = '../DatasetA_test_20180813/'
IMAGE_SIZE = 64
IMAGE_CHANNEL = 3
ALL_CLASSES = 230
ID_MAP = {}
i = 0
with open('../DatasetA_train_20180813/label_list.txt') as f:
    for line in f.readlines():
        id,_ = line.split()
        ID_MAP[id] = i
        i += 1
assert len(ID_MAP) == 230


def negsoftmax(arr):
    '''

    :param arr: np array
    :return: a newarray,first negtive it and then softmax
    '''
    # when implement the softmax ,it is crucial to first divide the max of the arr
    # to avoid the data overflow. but there is no need to do so because the
    # distance is postive and minus(arr) is negative and it wouldn't overflow
    # there will be a zero in each input arr,ignore it
    return np.where(arr,np.exp(-arr) / np.sum(np.exp(-arr)),0)


def constructDenseGraph():
    '''
    :return: a Dense Graph where each vertex is the word vector(300 dim), so the number of the
    vertex is the number of different classes of entities(including zero-shot entities and classes
    with enough training examples),and the link between nodes m,n is the influence the m gives n,it
    is a directed graph and the graph is not a symmetric graph because we will normalize each row.
    '''
    attributes_per_class_path = TRAIN_PATH_ROOT + 'attributes_per_class.txt'
    table_pd = pd.read_table(attributes_per_class_path, header=None)
    table_np = table_pd.as_matrix()[:,1:].astype(np.float64) # the numpy matrix of the attribute list
    class_ids = list(table_pd.as_matrix()[:,0])
    distance = {}
    for class_index in range(len(class_ids)):
        distance_temp = np.sum(np.square(table_np[class_index, :] - table_np), axis=1)
        distance_norm = negsoftmax(distance_temp)
        distance[class_ids[class_index]] = distance_norm
        # we need to normalize it!
    # first get the minus distance and softmax it
    G = nx.DiGraph()
    # read the word embeddings
    mapping_id_name, reverse_mapping = {},{}
    with open(TRAIN_PATH_ROOT + 'label_list.txt') as f:
        for line in f.readlines():
            id,class_name = line.split()
            mapping_id_name[id] = class_name
            reverse_mapping[class_name] = id
    embedding_path = TRAIN_PATH_ROOT + 'class_wordembeddings.txt'
    embedding_dict = {} # class id : numpy array of 300 dim (vector)
    with open(embedding_path) as f:
        for line in f.readlines():
            embedding_class_name, word_vec_str = line.split(' ', 1)
            word_vec_str_list = word_vec_str.split()
            word_vec_list = map(float,word_vec_str_list)
            word_vec_list_np = np.array(word_vec_list)
            if embedding_class_name not in reverse_mapping:
                print("error,class name not in label list(that should not happen!)")
                exit(1)
            embedding_dict[reverse_mapping[embedding_class_name]] = word_vec_list_np
    for class_id_t, word_vec_t in embedding_dict.items():
        G.add_node(class_id_t, word_vec=word_vec_t) # construct node and the word vector binding it
    print (G.nodes)
    print (G.number_of_nodes())
    for indexA in class_ids:
        for indexB_index in range(len(class_ids)):
            # add the edge between the two node
            G.add_weighted_edges_from([(indexA, class_ids[indexB_index], distance[indexA][indexB_index])])
    print (G.number_of_nodes())
    # test draw the draw function to draw the net
    return G


def cnn():
    '''
    :return: model (pretrained keras model)
    '''
    # we use cnn models pretrained from imagenet in keras
    model = VGG19(include_top=False,weights='imagenet',input_shape=(64,64,3))
    print(model.summary())
    return model # it will predict a 2 * 2 * 512 feature map from the training data,shape(2048,) if flattened()

def load_data_label(image_path='../DatasetA_train_20180813/train',label_path='../DatasetA_train_20180813/train.txt'):
    file_name_list = []
    train_size = len(os.listdir(image_path))
    label_dict = {}
    with open(label_path) as f:
        for line in f.readlines():
            file_name,label = line.split()
            if label not in ID_MAP:
                print('ERROR')
                exit(1)
            label_num = ID_MAP[label]
            label_dict[file_name] = K.one_hot(label_num,num_classes=ALL_CLASSES)
    data_shape = (train_size,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL)
    label_shape = (train_size,ALL_CLASSES)
    data = np.zeros(data_shape)
    label = np.zeros(label_shape)
    all_file_list = os.listdir(image_path)
    for file_name_index in range(len(all_file_list)):
        if all_file_list[file_name_index] not in label_dict:
            raise Exception('this file has no label(shouls not happen)')
        file_name = all_file_list[file_name_index]
        label[file_name_index,:] = label_dict[file_name]
        array_data = image.img_to_array(image.load_img(os.path.join(image_path,file_name)))
        data[file_name_index,:,:,:] = array_data
    return data, label

def train_val_test_split(X,y,ratio = [0.6,0.2],seed=42):
    # the ratio is a list of the ratio[train,test] ,and 1 - train_ratio - test_ratio is the validation set
    assert len(ratio) == 2
    assert sum(ratio) <= 1.0
    train_ratio, test_ratio , val_ratio = ratio[0],ratio[1],1 - ratio[0] - ratio[1]
    train_val_data,test_data,train_val_label,test_label = train_test_split(X,y,test_size=test_ratio,random_state=seed)
    train_data,train_label,val_data,val_lebel = train_test_split(train_val_data,train_val_label,test_size=val_ratio,random_state=seed)
    return train_data,val_data,

def train(g,data,label):
    # https://tkipf.github.io/graph-convolutional-networks/
    # it is a union train process where we use the pretrained model(VGG19) to extract features from
    # the image and get a 2048 dim (flattened) vector. and we train the gcn model with the input
    # (image_feature,wordvector matrix ,adjancy graph) and output the loss.
    '''

    :param g: graph
    :return: a new graph where each node contains the weight of this class
    '''
    weighted_adj_mat = nx.adjacency_matrix(g).toarray()
    feature_matrix = np.zeros(shape=(g.number_of_nodes(),g.node[0]['word_vec'].shape[0]))
    for node_index in range(len(g.node)):
        feature_matrix[node_index,:] = g.node[g.node[node_index]]['word_vec']
    # construct gcn model
    cnn_model = cnn()


    #weighted_adj_mat_tensor = K.constant(weighted_adj_mat)


if __name__ == '__main__':
    g = constructDenseGraph()
    #print (g.node['ZJL198'])
    print (g.get_edge_data('ZJL32','ZJL4')['weight'])
    # zjl1 - zjl111 0.00351107121949
    # zjl1 - zjl110 0.00107887753547
    # zjl111 - zjl123 0.00469994352486
    # zjl187 - zjl118 0.0132605135635
    # zjl187 - zjl189





