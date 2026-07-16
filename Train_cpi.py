import random
from Data_process import create_CPI_dataset
from utils import *
from metric import *
from Model_EMOE import PS_MoE
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, f1_score)
from datetime import datetime
from sklearn.metrics import confusion_matrix
import pandas as pd

CPIdatasets = ['DrugBank']
cuda_name = 'cuda:0'
ratio = 5
print('cuda_name:', cuda_name)
print('dataset:', CPIdatasets)
print('ratio', ratio)

'''set random seed'''
SEED = 3407
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

model_type = PS_MoE
TRAIN_BATCH_SIZE =  64
TEST_BATCH_SIZE =  64
LR = 0.0001
NUM_EPOCHS = 100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'Pretrain_Models'
results_dir = 'Pretrain_results'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(results_dir, CPIdatasets[0])):
    os.makedirs(os.path.join(results_dir, CPIdatasets[0]))
# 创建详细结果目录
detail_results_dir = os.path.join(results_dir, CPIdatasets[0], 'DrugBank')
if not os.path.exists(detail_results_dir):
    os.makedirs(detail_results_dir)

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = model_type()
model.to(device)
model_st = model_type.__name__
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for dataset in CPIdatasets:
    # train_data, valid_data, test_data = create_DTA_dataset(dataset)
    train_data, dev_data, test_data = create_CPI_dataset(type="warmup", dataset=dataset)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate, drop_last=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, drop_last=True)

    model_file_name = 'Pretrain_Models/DrugBank' + '.model'
    best_auc = 0
    best_epoch = 0
    for epoch in range(NUM_EPOCHS):
         train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)
         print('predicting for test data')
         Y, P, S, _, _ = predicting(model, device, test_loader)
         Val_Precision_dev = precision_score(Y, P)
         Val_Reacll_dev = recall_score(Y, P)
         Val_Accuracy_dev = accuracy_score(Y, P)
         Val_AUC_dev = roc_auc_score(Y, S)
         Val_F1 = f1_score(Y, P)
         tpr, fpr, _ = precision_recall_curve(Y, S)
         Val_PRC = auc(fpr, tpr)

         Y, P, S, _, _ = predicting(model, device, dev_loader)
         Dev_Precision_dev = precision_score(Y, P)
         Dev_Reacll_dev = recall_score(Y, P)
         Dev_Accuracy_dev = accuracy_score(Y, P)
         Dev_AUC_dev = roc_auc_score(Y, S)
         Dev_F1 = f1_score(Y, P)
         tpr, fpr, _ = precision_recall_curve(Y, S)
         Dev_PRC = auc(fpr, tpr)


         # 打印结果
         print(f'Test results - AUROC: {Val_AUC_dev:.4f}, Acc: {Val_Accuracy_dev:.4f}, '
               f'Precision: {Val_Precision_dev:.4f}, Recall: {Val_Reacll_dev:.4f}, '
               f'F1: {Val_F1:.4f}, AUPR: {Val_PRC:.4f}')
         print(f'Dev results - AUROC: {Dev_AUC_dev:.4f}, Acc: {Dev_Accuracy_dev:.4f}, '
               f'Precision: {Dev_Precision_dev:.4f}, Recall: {Dev_Reacll_dev:.4f}, '
               f'F1: {Dev_F1:.4f}, AUPR: {Dev_PRC:.4f}')


         save_file = os.path.join(results_dir, dataset, 'test_restult_' + str(ratio) + '_' + model_st + '.txt')
         open(save_file, 'w').writelines(result_str)
         if Dev_AUC_dev > best_auc :
             best_auc = Dev_AUC_dev
             best_epoch = epoch + 1
             torch.save(model.state_dict(), model_file_name)


    # test
    print('all training done. Testing...')
    model_p = model_type()
    model_p.to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    Y, P, S, _, _ = predicting(model_p, device, dev_loader)
    test_Precision_dev = precision_score(Y, P)
    test_Reacll_dev = recall_score(Y, P)
    test_Accuracy_dev = accuracy_score(Y, P)
    test_AUC_dev = roc_auc_score(Y, S)
    test_F1 = f1_score(Y, P)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    test_PRC = auc(fpr, tpr)

    # 计算混淆矩阵
    cm_dev = confusion_matrix(Y, P)
    tn_dev, fp_dev, fn_dev, tp_dev = cm_dev.ravel()

    # 创建最终结果字符串
    result_str = f'Best Model Results (Epoch {best_epoch}):\n'
    result_str += f'Test AUROC: {test_AUC_dev:.4f}\n'
    result_str += f'Test Accuracy: {test_Accuracy_dev:.4f}\n'
    result_str += f'Test Precision: {test_Precision_dev:.4f}\n'
    result_str += f'Test Recall: {test_Reacll_dev:.4f}\n'
    result_str += f'Test F1 Score: {test_F1:.4f}\n'
    result_str += f'Test AUPR: {test_PRC:.4f}\n'
    result_str += f'Confusion Matrix:\n'
    result_str += f'  True Positives (TP): {tp_dev}\n'
    result_str += f'  True Negatives (TN): {tn_dev}\n'
    result_str += f'  False Positives (FP): {fp_dev}\n'
    result_str += f'  False Negatives (FN): {fn_dev}\n'
    result_str += f'  Total Correct: {tp_dev + tn_dev}\n'
    result_str += f'  Total Wrong: {fp_dev + fn_dev}\n'

    print(result_str)

    # 保存最终结果
    save_file = os.path.join(results_dir, dataset, 'final_best_model_result_' + str(ratio) + '_' + model_st + '.txt')
    with open(save_file, 'w') as f:
        f.write(result_str)

