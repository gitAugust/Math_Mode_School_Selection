clc, clear
a=load('SMV.txt'); %�ѱ���x1...x8���������ݱ����ڴ��ı��ļ�fenlei.txt��
a=a(:,[1,3,5,7,8,9,10]);
%[2,3,5,7,8,9]
nu=500;
f=a(1:nu,:);
t=a(501:(nu+500),:);
c=[f;t];
b0=c((1:(2*nu)),:); 
dd0=a((1001:2000),:); %��ȡ�ѷ���(b0)��1��20�����Ըģ��ʹ����������(dd0)
% [train_data,ps]=mapstd(b0); %�ѷ������ݵı�׼��
% test_data=mapstd('apply',dd0,ps); %���������ݵı�׼��
train_data=b0;
test_data=dd0;
train_label=[-1*ones(nu,1);ones(nu,1)]; %��֪������������(ǰ��10����1����10����2��һ��ǰΪ�棬��Ϊ�٣�


%save label.mat;%����Ϊ������
pr=train_data';%��ֵ
tr1=train_label';
ar=test_data';
% [pr1,ps]=mapminmax(pr);%��һ��������Χ-1~1������ֵ����test1����һ������ts
% ar1=mapminmax('apply',ar,ps); %���������ݵı�׼��
pr1=pr;
ar1=ar;
net=newff(pr1,tr1,14);%�������磬���㾭�鹫ʽ9-17
net.trainParam.epochs=10000;
net.trainParam.goal=1e-7;
net.trainParam.lr=0.01;%ѧϰ�� 
net.trainParam.mc=0.9;%������������1
net.trainParam.show=25;%��ʾ�ļ������
[net,tr]=train(net,pr1,tr1);%ѵ��������
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