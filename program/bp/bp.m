%ͨ�����������㷨
clear,clc

tr=load('train_data.txt');
%train_data.txt��ʽ��ǰ��ǧ��Ϊtrue�Ĳ�������ǧ��Ϊfake����
%                    ÿ��Ϊһ��ͼ�� ÿ��Ϊһ������
val=load('validate_data.txt')';
%validate_data.txt��ʽ����һǧ��ͼ��Ĳ���
%                       ÿ��Ϊһ��ͼ�� ÿ��Ϊһ������
tes=load('test.txt');
%test.txt��ʽ����һǧ��ͼ��Ĳ���
%              ÿ��Ϊһ��ͼ�� ÿ��Ϊһ������

%������ ���� ѵ�� ����
net1=bp_cat(tr,tes,8000);
S=sim(net1,val);

S(find(S<=0))=0;
S(find(S>0))=1;

%д����
xlswrite('val_result.xlsx',S');