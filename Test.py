import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from DLBCLdataset import myNumpyDataset
# from Model import LymRadar
from sklearn.metrics import recall_score, roc_auc_score, roc_curve, auc,  accuracy_score
import matplotlib.pyplot as plt
from visdom import Visdom
from tqdm import tqdm
import random
import pandas as pd
import torch.nn.functional as F
import SimpleITK as sitk

def getData(datadir,Label):
    dataset = []
    dataset1 = []
    dataset2 = []

    subforder_list = os.listdir(datadir)
    for allnames in subforder_list:
        name1_path = os.path.join(datadir,allnames,'CT.npy')
        name2_path = os.path.join(datadir,allnames,'PET.npy')
        for lb in Label:
            if lb[0] == allnames:
                dataset1.append((int(lb[1]),name1_path))
                dataset2.append((int(lb[1]),name2_path))
    dataset.append(dataset1)
    dataset.append(dataset2)
    return dataset, dataset1, dataset2

def PETCTConLoss(feature_a, feature_b, pred_a, pred_b, labels, temperature=0.05):
    feature_a = F.normalize(feature_a, dim=1)
    feature_b = F.normalize(feature_b, dim=1)

    correct_a = pred_a.eq(labels)
    correct_b = pred_b.eq(labels)

    both_correct = correct_a & correct_b      # TT
    both_wrong = (~correct_a) & (~correct_b)  # FF
    disagree = correct_a ^ correct_b          # TF / FT

    loss_pull = torch.tensor(0., device=feature_a.device)
    loss_align = torch.tensor(0., device=feature_a.device)
    loss_repulse = torch.tensor(0., device=feature_a.device)

    valid_terms = 0

    if both_correct.any():
        sim = F.cosine_similarity(
            feature_a[both_correct],
            feature_b[both_correct]
        )

        loss_pull = (1 - sim).mean() / temperature
        valid_terms += 1

    if disagree.any():
        a_fixed = feature_a.clone()
        b_fixed = feature_b.clone()

        # detach 正确模态
        a_fixed[correct_a & disagree] = feature_a[correct_a & disagree].detach()
        b_fixed[correct_b & disagree] = feature_b[correct_b & disagree].detach()

        sim = F.cosine_similarity(
            a_fixed[disagree],
            b_fixed[disagree]
        )

        loss_align = (1 - sim).mean() / temperature
        valid_terms += 1

    if both_wrong.any():
        sim = F.cosine_similarity(
            feature_a[both_wrong],
            feature_b[both_wrong]
        )

        repulse = F.relu(sim)

        loss_repulse = repulse.mean()
        valid_terms += 1

    if valid_terms == 0:
        return torch.tensor(0., requires_grad=True, device=feature_a.device)

    total_loss = ( loss_pull + loss_align + loss_repulse   ) / valid_terms

    return total_loss


def Test(root1,datalist1,datalist2,weight):

    savepath = '***'
    savename = '***'
    result_out_path = '/***/***'
    mydata = myNumpyDataset(root_dir=root1, img_list1=datalist1, img_list2=datalist2, transform='test')
    mydataloader = DataLoader(mydata, batch_size=1, shuffle=False, num_workers=8)

    model = torch.load('LymRadarModelTrained.pt', map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.CrossEntropyLoss()
    loss3 = nn.CrossEntropyLoss()
    loss4 = PETCTConLoss

    loss_listtest = []
    test_y_hattest = []
    test_y_truetest = []
    test_y_pertest = []
    test_name = []
    saveFeature_test = []


    with torch.no_grad():
        print('********** start test **********')
        lossi_test = 0
        model.eval()
        for testbatchimgs1, testbatchlabels1, testbatchimgs2, testbatchlabels2, testname in tqdm(mydataloader):
            testimgs1 = testbatchimgs1.to(mydevice)
            testlabels1 = testbatchlabels1.to(mydevice)
            testimgs2 = testbatchimgs2.to(mydevice)

            x1, x2, output_ct, output_pet, output_mm, outsave_features_test = model(testimgs1, testimgs2)
            bloss1 = loss1(output_ct, testlabels1)
            bloss2 = loss2(output_pet, testlabels1)
            bloss3 = loss3(output_mm, testlabels1)
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            x2 = x2 / x2.norm(dim=-1, keepdim=True)
            output_ct = torch.argmax(F.softmax(output_ct, dim=1), dim=1)
            output_pet = torch.argmax(F.softmax(output_pet, dim=1), dim=1)
            bloss4 = loss4(x1, x2, output_ct, output_pet, testlabels1)
            bloss = bloss1 + bloss2 + bloss3 + 0.1 * bloss4
            lossi_test = lossi_test + bloss.item() * 1  ### 每个batch的loss，叠加汇总成一个epoch的loss

            _, predictedtest = torch.max(output_mm.data, 1)  # 预测的类别分类
            predictedtest = predictedtest.cpu().numpy()
            pred_y_softmaxtest = torch.softmax(output_mm, dim=1).detach().cpu().numpy()[:, 1]  # 预测的分类概率
            outsave_features_test = outsave_features_test.cpu().numpy()
            test_y_hattest.extend(list(predictedtest))
            test_y_truetest.extend(list(testlabels1.cpu().numpy()))
            test_y_pertest.extend(list(pred_y_softmaxtest))
            test_name.extend(list(np.array(testname)))
            saveFeature_test.extend(list(np.array(outsave_features_test)))

        test_y_hattest = np.array(test_y_hattest).reshape(-1)
        test_y_truetest = np.array(test_y_truetest).reshape(-1)
        test_y_pertest = np.array(test_y_pertest).reshape(-1)


        new_test_y_hattest = np.hstack((test_name, test_y_hattest))
        new_test_y_truetest = np.hstack((test_name, test_y_truetest))
        new_test_y_pertest = np.hstack((test_name, test_y_pertest))

        np.savetxt('{}/{}/{}-predProbability.txt'.format(result_out_path,savepath,savename), new_test_y_pertest,fmt='%s')

        loss_listtest.append(lossi_test)
        print('loss_test:{}'.format(loss_listtest[-1] / len(mydataloader)))

        # 绘制ROC曲线，计算AUC
        fprtest, tprtest, thresholds_roctest = roc_curve(test_y_truetest, test_y_pertest)
        AUCtest = auc(fprtest, tprtest)

        ACCtest = accuracy_score(y_true=test_y_truetest, y_pred=test_y_hattest)
        print('AUCtest', AUCtest, 'ACCtest', ACCtest)


if __name__ == '__main__':
        labels = np.loadtxt('Label.txt',dtype=str)
        dataset_root1 = 'NpyData'
        dataset, dataset1, dataset2 = getData(dataset_root1,labels)
        Test(root1=dataset_root1,datalist1=dataset1,datalist2=dataset2, weight='LymRadar.pth')
