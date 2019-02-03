function [net,r,s] = bp_cat(tr,test_data,nu0)
%bp_cat ������ѵ��������
%���룺tr   ���� ÿ��Ϊһ��ͼ�� ÿ��Ϊһ������
%           ǰ��ǧ��Ϊtrue�Ĳ�������ǧ��Ϊfake����
%       nu0 ��������
% �����net ���纯��
%       r   ׼ȷ��
%       s   ��ȷʶ�������test������Ϊ1000��

nu=nu0/2;%nu=������/2
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
train_label=[ones(nu,1);-1*ones(nu,1)]; %train������������

tr1=train_data';%��ֵ
pr1=train_label';
ar1=test_data';
test_label=[ones(500,1);-1*ones(500,1)]; %train������������

net=newff(tr1,pr1,13,{'tansig','purelin'});%�������磬���㾭�鹫ʽ9-17
net.trainParam.epochs=10;
net.trainParam.goal=1e-7;
net.trainParam.lr=0.01;%ѧϰ�� 
net.trainParam.mc=0;%������������1
net.trainParam.show=25;%��ʾ�ļ������bp
[net,tr]=train(net,tr1,pr1);%ѵ��������
predict_label=sim(net,ar1);



%��ȷ�ʼ��
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

%s:��ȷ�ĸ��� r
s=m+n
r=s/10

%��ͼ_Ԥ������ǩ��ʵ�ʷ����ǩ�Ա�ͼ
figure(1);
plot(predict_label,'b*')
hold on
plot(test_label,'r');
hold on
plot(0:1000,0,'bl');
legend('ʶ���ǩ','ʵ�ʱ�ǩ')
title('BP�����������ʵ�����ȶ�','fontsize',12)
ylabel('����ǩ','fontsize',12)
xlabel('������Ŀ','fontsize',12)
ylim([-1.5 1.5])
end



