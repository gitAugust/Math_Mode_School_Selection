clc, clear
a0=load('SMV.txt'); %�ѱ���x1...x8���������ݱ����ڴ��ı��ļ�fenlei.txt��
a=a0'; b0=a(:,(1:20)); dd0=a(:,(21:end)); %��ȡ�ѷ���(b0)��1��20�����Ըģ��ʹ����������(dd0)
[b,ps]=mapstd(b0); %�ѷ������ݵı�׼��
dd=mapstd('apply',dd0,ps); %���������ݵı�׼��
group=[ones(10,1); 2*ones(10,1)]; %��֪������������(ǰ��10����1����10����2��һ��ǰΪ�棬��Ϊ�٣�
s=fitcsvm(b',group); %ѵ��֧��������������
sv_index=s.SupportVectorIndices  %����֧�������ı��
beta=s.Alpha  %���ط��ຯ����Ȩϵ��
bb=s.Bias  %���ط��ຯ���ĳ�����
mean_and_std_trans=s.ScaleData %��1�з��ص�����֪�������ֵ�������෴������2�з��ص��Ǳ�׼�������ĵ���
check=ClassificationSVM(s,b')  %��֤��֪������
err_rate=1-sum(group==check)/length(group) %������֪������Ĵ�����
solution=ClassificationSVM(s,dd') %�Դ�����������з���
