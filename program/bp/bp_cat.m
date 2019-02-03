function [net,r,s] = bp_cat(tr,test_data,nu0)
%bp_cat 创建并训练神经网络
%输入：tr   矩阵 每行为一个图像 每列为一个参数
%           前五千个为true的参数后五千个为fake参数
%       nu0 样本个数
% 输出：net 网络函数
%       r   准确率
%       s   正确识别个数（test总数量为1000）

nu=nu0/2;%nu=样本数/2
f=tr(1:nu,:);
t=tr(5001:(nu+5000),:);
c=[f;t];

train_data=c((1:end),:); 
%[train_data] = premnmx(train_data0);
% 
% tes=load('test.txt')';
% test_data=tes'; 
%[test_data] = premnmx(test_data0);

%test_data=tr((4501:5500),:); 
train_label=[ones(nu,1);-1*ones(nu,1)]; %train样本点的类别标号

tr1=train_data';%赋值
pr1=train_label';
ar1=test_data';
test_label=[ones(500,1);-1*ones(500,1)]; %train样本点的类别标号

net=newff(tr1,pr1,13,{'tansig','purelin'});%创建网络，隐层经验公式9-17
net.trainParam.epochs=10;
net.trainParam.goal=1e-7;
net.trainParam.lr=0.01;%学习率 
net.trainParam.mc=0;%动量因子设置1
net.trainParam.show=25;%显示的间隔次数bp
[net,tr]=train(net,tr1,pr1);%训练神经网络
predict_label=sim(net,ar1);



%正确率检测
m=0;
n=0;
for i=1:500
    if predict_label(1,i)>0
        m=m+1;
    end
    if predict_label(1,i+500)<0
        n=n+1;
    end
end

%s:正确的个数 r
s=m+n
r=s/10

%制图_预测分类标签和实际分类标签对比图
figure(1);
plot(predict_label,'b*')
hold on
plot(test_label,'r');
hold on
plot(0:1000,0,'bl');
legend('识别标签','实际标签')
title('BP神经网络分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
ylim([-1.5 1.5])
end



