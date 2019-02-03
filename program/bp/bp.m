%通过参数分类算法
clear,clc

tr=load('train_data.txt');
%train_data.txt格式：前五千个为true的参数后五千个为fake参数
%                    每行为一个图像 每列为一个参数
val=load('validate_data.txt')';
%validate_data.txt格式：共一千个图像的参数
%                       每行为一个图像 每列为一个参数
tes=load('test.txt');
%test.txt格式：共一千个图像的参数
%              每行为一个图像 每列为一个参数

%神经网络 创建 训练 分类
net1=bp_cat(tr,tes,8000);
S=sim(net1,val);

S(find(S<=0))=0;
S(find(S>0))=1;

%写入表格
xlswrite('val_result.xlsx',S');