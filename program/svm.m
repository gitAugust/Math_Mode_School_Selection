clc, clear
a0=load('SMV.txt'); %把表中x1...x8的所有数据保存在纯文本文件fenlei.txt中
a=a0'; b0=a(:,(1:20)); dd0=a(:,(21:end)); %提取已分类(b0)（1至20，可以改）和待分类的数据(dd0)
[b,ps]=mapstd(b0); %已分类数据的标准化
dd=mapstd('apply',dd0,ps); %待分类数据的标准化
group=[ones(10,1); 2*ones(10,1)]; %已知样本点的类别标号(前面10个是1，后10个是2，一般前为真，后为假）
s=fitcsvm(b',group); %训练支持向量机分类器
sv_index=s.SupportVectorIndices  %返回支持向量的标号
beta=s.Alpha  %返回分类函数的权系数
bb=s.Bias  %返回分类函数的常数项
mean_and_std_trans=s.ScaleData %第1行返回的是已知样本点均值向量的相反数，第2行返回的是标准差向量的倒数
check=ClassificationSVM(s,b')  %验证已知样本点
err_rate=1-sum(group==check)/length(group) %计算已知样本点的错判率
solution=ClassificationSVM(s,dd') %对待判样本点进行分类
