import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import sys
import matplotlib.pyplot as plt
from models import cnn, resnet, 
from UCI_HAR.dataproc import UCI
from models.dis_losses.kl_div import KLDivergence
from models.dis_losses.RRKD import RRKD
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from tqdm import tqdm
from utils import save_arrays,load_arrays
import pickle
import pandas as pd
import random

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed(0)

# 直接设置参数

dataset_name = 'uci'
model_type = 'resnet'

savepath = '../HAR-datasets'
batch_size = 256
epoch = 300
lr = 0.005


save_folder = 'epoch_acc'



dataset_dict = {
    'uci': UCI,
    'unimib': UNIMIB,
    'wisdm': WISDM,
    'oppo': OPPO,
    'pamap': PAMAP
}
dir_dict = {
    'uci': 'UCI_HAR',
    'unimib': 'UniMiB_SHAR',
    'wisdm': 'WISDM',
    'pamap': 'PAMAP2',
    'oppo': 'OPPORTUNITY'
}

model_dict = {
    'cnn': cnn.CNN, 
    'cnns': cnn.CNN8_S,
    'resnet': resnet.ResNet,
    'res2net': res2net.Res2Net,
    'resnext': resnext.ResNext,
}

savepath = os.path.abspath(savepath) if savepath else ''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_data(dataset_name):
    '''数据集加载,产生训练和测试数据，并加载教师网络'''
    #print('\n==================================================【数据集预处理】===================================================\n')
    dataset_instance = dir_dict[dataset_name]
    dataset_saved_path = os.path.join(savepath, str(dataset_instance))




    '''获取训练与测试【数据，标签】'''
    train_data, test_data, train_label, test_label = np.load('%s/x_train.npy'%(dataset_saved_path)), np.load('%s/x_test.npy'%(dataset_saved_path)), np.load('%s/y_train.npy'%(dataset_saved_path)), np.load('%s/y_test.npy'%(dataset_saved_path))

    '''npy数据tensor化'''

    X_train = torch.from_numpy(train_data).float().unsqueeze(1)
    X_test = torch.from_numpy(test_data).float().unsqueeze(1)
    Y_train = torch.from_numpy(train_label).long()
    Y_test = torch.from_numpy(test_label).long()

    category = len(set(Y_test.tolist()))

    '''模型加载'''
    print('\n==================================================  【模型加载】  ===================================================\n')
    

    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    net_Teacher = model_dict[model_type](X_test.shape, category).to(device)
    load_teacher_model(net_Teacher, dataset_name, model_type)
    
    return X_train,X_test,Y_train,Y_test,category,train_loader,test_loader,net_Teacher
    
'''加载教师网络并训练学生网络'''

def load_teacher_model(model, dataset, model_type):
    # Load the pre-trained teacher model
    path_to_teacher_model = f'./teacher_{dataset}_resnet.pth'
    checkpoint = torch.load(path_to_teacher_model)
    model.load_state_dict(checkpoint)
    model.eval()



X_train,X_test,Y_train,Y_test,category,train_loader,test_loader,net_Teacher = generate_data(dataset_name)


def train_and_evaluate_model(model_s,dataset_name):

    loss_fn_RRKD = RRKD(tau=2)
    loss_fn = nn.CrossEntropyLoss()
    scaler_stu = GradScaler() # 在训练最开始之前实例化一个GradScaler对象
    optimizer_stu = torch.optim.AdamW(model_s.parameters(), lr=lr, weight_decay=0.001)
    loss_fn_RRKD = torch.optim.lr_scheduler.StepLR(optimizer_stu, epoch//3, 0.5)
    scaler_stu = GradScaler() # 在训练最开始之前实例化一个GradScaler对象
    best_val_acc_stu = 0
    for i in tqdm(range(epoch)):
        model_s.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            # 前向过程(model + loss)开启 autocast，混合精度训练
            with autocast():
                out_T = net_Teacher(data)           #教师输出
                out_S = model_s(data)           #学生输出      
                Lcls = loss_fn(out_S, label)  # 分类损失
                Linter, Lintra = loss_fn_RRKD(out_S, out_T)                       
                loss = Lcls + 2 * Linter + 4 * Lintra

            optimizer_stu.zero_grad() # 梯度清零
            scaler_stu.scale(loss).backward() # 梯度放大
            scaler_stu.step(optimizer_stu) # unscale梯度值
            scaler_stu.update() 

        # 在每个epoch结束后进行验证
        model_s.eval()
        cor = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            with autocast():
                out = model_s(data)
            _, pre = torch.max(out, 1)
            cor += (pre == label).sum()
        acc = cor.item() / len(Y_test)

        
        # 调用函数保存数组，并传递 dataset 和 model 参数
        save_arrays(epoch_list_s_dist, acc_list_s_dist, dataset=dataset_name, model=model_type,epoch_file_name='epoch_student_{}'.format(dataset_name),acc_file_name='acc_student_{}'.format(dataset_name))             
        if (acc > best_val_acc_stu ):
            best_val_acc_stu = acc
            path_s = 'model_params2/student_{}_RRKD.pth'.format(dataset_name)
            torch.save(model_s.state_dict(), path_s)
        if((i+1)%50==0):
            print('epoch: %d, train-loss: %f, val-acc: %f, best-val-acc: %f ' % (i, loss, acc, best_val_acc_stu))

    acc_all.append(average_last_20_accs)
    return acc,best_val_acc_stu

def load_checkpoint(filename):
    """如果存在checkpoint文件，则加载该文件"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def evaluate_model(model, test_loader):
    """评估模型在测试集上的性能，并返回准确率、精确率、召回率和 F1 值"""
    model.eval()  # 将模型设置为评估模式
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
      
    accuracy = accuracy_score(true_labels, predicted_labels)  
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    weighted_f1 = report['weighted avg']['f1-score']  

    return accuracy, precision, recall, f1, weighted_f1

def print_evaluation_results(dataset_name, model_s, test_loader,teach = False):
    """
    打印模型在测试集上的评估结果。
    
    参数:
    - dataset_name: 数据集名称
    - model_s: 学生模型名称
    - test_loader: 测试集 DataLoader
    - teach 为true就加载教师网络
    """
    # 加载学生网络权重
    if teach == False:
        model_s.load_state_dict(torch.load('model_params/student_{}_RRKD.pth'.format(dataset_name)))
    else:
        model_s.load_state_dict(torch.load('model_params/teacher_{}_resnet.pth'.format(dataset_name)))
    # 评估模型性能
    accuracy, precision, recall, f1, weighted_f1 = evaluate_model(model_s, test_loader)


    df = pd.DataFrame({
        'Dataset': [dataset_name] + [''] * 4,
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] + [''],
        'Value': [accuracy, precision, recall, f1] + ['']
    })

    print(df.to_string(index=False))
    print(f"Weighted F1: {weighted_f1}")

def train_and_evaluate_model_vanillaKD(model_s,dataset_name):

    optimizer_stu = torch.optim.AdamW(model_s.parameters(), lr=lr, weight_decay=0.001)
    lr_sch_stu_dist = torch.optim.lr_scheduler.StepLR(optimizer_stu, epoch//3, 0.5)
    loss_fn_kl = KLDivergence(tau=2)

    loss_fn = nn.CrossEntropyLoss()

    scaler_stu = GradScaler() # 在训练最开始之前实例化一个GradScaler对象

    best_val_acc_stu = 0
    for i in tqdm(range(epoch)):
        model_s.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            # 前向过程(model + loss)开启 autocast，混合精度训练
            with autocast():
                out_T = net_Teacher(data)           #教师输出
                out_S = model_s(data)           #学生输出
                loss = loss_fn_kl(out_S, out_T) + loss_fn(out_S,label)            

            optimizer_stu.zero_grad() # 梯度清零
            scaler_stu.scale(loss).backward() # 梯度放大
            scaler_stu.step(optimizer_stu) # unscale梯度值
            scaler_stu.update() 

        model_s.eval()
        cor = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            with autocast():
                out = model_s(data)
            _, pre = torch.max(out, 1)
            cor += (pre == label).sum()
        acc = cor.item() / len(Y_test)     
        if (acc > best_val_acc_stu ):
            best_val_acc_stu = acc
            path_s = 'model_params/student_{}_vaniKD.pth'.format(dataset_name)
            torch.save(model_s.state_dict(), path_s)
        if((i+1)%1==0):
            print('epoch: %d, train-loss: %f, val-acc: %f, best-val-acc: %f ' % (i, loss, acc, best_val_acc_stu)) 
    
    model_s.load_state_dict(torch.load(path_s))
    accuracy, precision, recall, f1 ,weighted_f1= evaluate_model(model_s,test_loader)
    df = pd.DataFrame({
    'Dataset': [dataset_name] + [''] * 4,
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'] + [''],
    'Value': [accuracy, precision, recall, f1] + ['']
    })
    print('vanilla KD \r\n')
    print(df.to_string(index=False))
    print(f"Weighted F1: {weighted_f1}")

def task(model_s,dataset_name):
    X_train,X_test,Y_train,Y_test,category,train_loader,test_loader,net_Teacher=generate_data(dataset_name)
    accuracy, best_val_acc = train_and_evaluate_model(model_s,dataset_name)
    print_evaluation_results(dataset_name,model_s,test_loader)

#训练dist
epoch = 300

dataset_name = 'uci'
net_sota = cnn.CNN8_S(X_train.shape, category).to(device)
train_and_evaluate_model_student(net_sota,dataset_name)



