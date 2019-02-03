clc, clear
a=load('SMV.txt'); %把表中x1...x8的所有数据保存在纯文本文件fenlei.txt中
a=a(:,[1,3,5,7,8,9,10]);
%[2,3,5,7,8,9]
nu=500;
f=a(1:nu,:);
t=a(501:(nu+500),:);
c=[f;t];
b0=c((1:(2*nu)),:); 
dd0=a((1001:2000),:); %提取已分类(b0)（1至20，可以改）和待分类的数据(dd0)
% [train_data,ps]=mapstd(b0); %已分类数据的标准化
% test_data=mapstd('apply',dd0,ps); %待分类数据的标准化
train_data=b0;
test_data=dd0;
train_label=[-1*ones(nu,1);ones(nu,1)]; %已知样本点的类别标号(前面10个是1，后10个是2，一般前为真，后为假）


%save label.mat;%必须为行向量
pr=train_data';%赋值
tr1=train_label';
ar=test_data';
% [pr1,ps]=mapminmax(pr);%归一化处理，范围-1~1，返回值数据test1，归一化参数ts
% ar1=mapminmax('apply',ar,ps); %待分类数据的标准化
pr1=pr;
ar1=ar;
net=newff(pr1,tr1,14);%创建网络，隐层经验公式9-17
net.trainParam.epochs=10000;
net.trainParam.goal=1e-7;
net.trainParam.lr=0.01;%学习率 
net.trainParam.mc=0.9;%动量因子设置1
net.trainParam.show=25;%显示的间隔次数
[net,tr]=train(net,pr1,tr1);%训练神经网络
predict_label=sim(net,ar1)


m=0;
n=0;
for i=1:500
    if predict_label(1,i)<0
        m=m+1;
    end
    if predict_label(1,i+500)>0
        n=n+1;
    end
end

s=m+n
r=s/100