clear
clc
I=imread('5.jpeg');%path�����·��
A=imread('1.jpeg');

I_R=I(:,:,1);%R����
I_G=I(:,:,2);%G����
I_B=I(:,:,3);%B����

A_R=A(:,:,1);%R����
A_G=A(:,:,2);%G����
A_B=A(:,:,3);%B����

[a,d]=mdwt(I_R-I_G);

%%����ͼ��������NRSS
%%reference paper:һ�����ͼ��ģ�����޲ο���������ָ�꣬�����Ӧ�ã�лС���ȡ�

img = a;  %��ȡԭʼͼ��
N = 64;  %ȡ��������ǰN��
block_size = 8;  %���С
stride = 4;  %�ֿ鲽����С��block_sizeʱ���ص����֣�
 
%%��ʼ��
blk_count = 0;
ssim_sum = 0;
G_blk = zeros(block_size,block_size,1);
Gr_blk = zeros(block_size,block_size,1);
G_std = zeros(1);
 
%% (1)��ͨ�˲�,���ɲο�ͼ��Ir
sigma = sqrt(6);
if size(img,3) == 3
    img = rgb2gray(img);  %�ҶȻ�
end
[m,n] = size(img);
gausFilter = fspecial('gaussian',[7 7],sigma);  %������˹�˲���
Ir = imfilter(img,gausFilter,'replicate');  %��˹�˲�
figure,
subplot(121),imshow(img),title('Original Image');
subplot(122),imshow(Ir),title('Gaussian Filter Image');
 
%% (2)����Sobel���Ӽ���ͼ��img��Ir���ݶ�ͼ��G��Gr
G = edge(img,'sobel');  %��Sobel�������ݶ�ͼ��
Gr= edge(Ir,'sobel');
figure,
subplot(121),imshow(G),title('G');
subplot(122),imshow(Gr),title('Gr');
 
%% (3)���ݶ�ͼ�񻮷ֳ�С�鲢����ÿ��ķ���ҳ����з�������ǰN��
for i = 1:stride:m-block_size+1
    for j = 1:stride:n-block_size+1
        blk_count = blk_count+1;
        G_blk(:,:,blk_count) = G(i:i+block_size-1,j:j+block_size-1);
        Gr_blk(:,:,blk_count) = Gr(i:i+block_size-1,j:j+block_size-1);
    end
end
if blk_count <= N
    N = blk_count;
end
 
for i = 1:blk_count
    G_std(i) = (std2(G_blk(:,:,i)))^2;  %���㷽��
end
G_std_sort = sort(G_std,'descend');  %���������
G_indice = find(G_std >= G_std_sort(N));
G_indice = G_indice(1:N);  %ȡǰN�����±�
 
%% (4)����ͼ����޲ο��ṹ������
for i = G_indice
    mssim = ssim(G_blk(:,:,i),Gr_blk(:,:,i));
    ssim_sum = ssim_sum+mssim;
end
nrss = 1-ssim_sum/N;%���ù�ʽ����õ�NRSS
display(nrss);
% eda = edge(a, 'canny', 0.5); 
% edb = edge(b, 'canny', 0.5); 
% edc = edge(c, 'canny', 0.5); 
% figure(3)    
%     subplot(4,2,1);
%         imshow(a);%ԭͼ
%     subplot(4,2,2);
%         imshow(eda);%��˹�˲���
%     subplot(4,2,3);
%         imshow(b);%����
%     subplot(4,2,4);
%         imshow(edb);%�Ǽ���ֵ����
%     subplot(4,2,5);
%         imshow(c);%˫��ֵ
%     subplot(4,2,6);
%         imshow(edc);%Matlab�Դ���Ե���

% function [LL,HL,LH,HH] = mdwt(A)
% %UNTITLED2 �˴���ʾ�йش˺�����ժҪ
% %   �˴���ʾ��ϸ˵��
% 
% f=A(1:374,1:374);%�Ѿ���任Ϊ����һ����Ϊż���ľ���
% d=size(f);
% if length(d)>2
%     f=rgb2gray((f));%%%%%%%%����ǲ�ɫͼ����ת��Ϊ�Ҷ�ͼ
% end
% T=d(1);
% SUB_T=T/2;
% %  2.���ж�άС���ֽ�
% l=wfilters('db10','l');    %  db10����ʧ��Ϊ10)��ͨ�ֽ��˲��������Ӧ������Ϊ20��
% L=T-length(l);
% l_zeros=[l,zeros(1,L)];    %  ��������������ͼ��һ�£�Ϊ2��������
% h=wfilters('db10','h');    %  db10����ʧ��Ϊ10)��ͨ�ֽ��˲��������Ӧ������Ϊ20��
% h_zeros=[h,zeros(1,L)];    %  ��������������ͼ��һ�£�Ϊ2��������
% for i=1:T   %  �б任
%     row(1:SUB_T,i)=dyaddown( ifft( fft(l_zeros).*fft(f(:,i)') ) ).';    %  Բ�ܾ��<->FFT
%     row(SUB_T+1:T,i)=dyaddown( ifft( fft(h_zeros).*fft(f(:,i)') ) ).';  %  Բ�ܾ��<->FFT
% end
% for j=1:T   %  �б任
%     line(j,1:SUB_T)=dyaddown( ifft( fft(l_zeros).*fft(row(j,:)) ) );    %  Բ�ܾ��<->FFT
%     line(j,SUB_T+1:T)=dyaddown( ifft( fft(h_zeros).*fft(row(j,:)) ) );  %  Բ�ܾ��<->FFT
% end
% decompose_pic=line;  %  �ֽ����
% %  ͼ���Ϊ�Ŀ�
% lt_pic=decompose_pic(1:SUB_T,1:SUB_T);      %  �ھ������Ϸ�Ϊ��Ƶ����--fi(x)*fi(y)
% rt_pic=decompose_pic(1:SUB_T,SUB_T+1:T);    %  ��������Ϊ--fi(x)*psi(y)
% lb_pic=decompose_pic(SUB_T+1:T,1:SUB_T);    %  ��������Ϊ--psi(x)*fi(y)
% rb_pic=decompose_pic(SUB_T+1:T,SUB_T+1:T);  %  ���·�Ϊ��Ƶ����--psi(x)*psi(y)
%  
% %  3.�ֽ�����ʾ
% figure(1);
% subplot(2,1,1);
% imshow(f,[]);  %  ԭʼͼ��  
% title('original pic');
% subplot(2,1,2);
% image(abs(decompose_pic));  %  �ֽ��ͼ��
% title('decomposed pic');
% figure(2);
% % colormap(map);
% LL=lt_pic;
% subplot(2,2,1);
% imshow(abs(lt_pic),[]);  %  ���Ϸ�Ϊ��Ƶ����--fi(x)*fi(y)
% title('\Phi(x)*\Phi(y)');
% subplot(2,2,2);
% LH=rt_pic;
% imshow(abs(rt_pic),[]);  %  ��������Ϊ--fi(x)*psi(y)
% title('\Phi(x)*\Psi(y)');
% HL=lb_pic;
% subplot(2,2,3);
% imshow(abs(lb_pic),[]);  %  ��������Ϊ--psi(x)*fi(y)
% title('\Psi(x)*\Phi(y)');
% HH=rb_pic;
% subplot(2,2,4);
% imshow(abs(rb_pic),[]);  %  ���·�Ϊ��Ƶ����--psi(x)*psi(y)
% title('\Psi(x)*\Psi(y)');
%  
% %  5.�ع�Դͼ�񼰽����ʾ
% % construct_pic=decompose_matrix'*decompose_pic*decompose_matrix;
% % l_re=l_zeros(end:-1:1);   %  �ع���ͨ�˲�
% % l_r=circshift(l_re',1)';  %  λ�õ���
% % h_re=h_zeros(end:-1:1);   %  �ع���ͨ�˲�
% % h_r=circshift(h_re',1)';  %  λ�õ���
% % top_pic=[lt_pic,rt_pic];  %  ͼ���ϰ벿��
% % t=0;
% % for i=1:T  %  �в�ֵ��Ƶ
% %  
% %     if (mod(i,2)==0)
% %         topll(i,:)=top_pic(t,:); %  ż���б���
% %     else
% %         t=t+1;
% %         topll(i,:)=zeros(1,T);   %  ������Ϊ��
% %     end
% % end
% % for i=1:T  %  �б任
% %     topcl_re(:,i)=ifft( fft(l_r).*fft(topll(:,i)') )';  %  Բ�ܾ��<->FFT
% % end
% %  
% % bottom_pic=[lb_pic,rb_pic];  %  ͼ���°벿��
% % t=0;
% % for i=1:T  %  �в�ֵ��Ƶ
% %     if (mod(i,2)==0)
% %         bottomlh(i,:)=bottom_pic(t,:);  %  ż���б���
% %     else
% %         bottomlh(i,:)=zeros(1,T);       %  ������Ϊ��
% %         t=t+1;
% %     end 
% % end
% % end



%%%%%%%%%%%%%%bp%%%%%%%%%%%%%%
clc, clear

tr=load('train_data.txt');
te=load('test_data.txt');

a=load('SMV.txt'); %�ѱ���x1...x8���������ݱ����ڴ��ı��ļ�fenlei.txt��
a=a(:,[1,3,5,7,8,9,10]);
%[2,3,5,7,8,9]
nu=500;
f=tr(1:nu,:);
t=tr(5001:(nu+5000),:);
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


%�����ͼ
%��ͼ_Ԥ������ǩ��ʵ�ʷ����ǩ�Ա�ͼ
figure(1)
plot(y,'og')
hold on
plot(label_test,'r*');
legend('Ԥ���ǩ','ʵ�ʱ�ǩ')
title('BP������Ԥ�������ʵ�����ȶ�','fontsize',12)
ylabel('����ǩ','fontsize',12)
xlabel('������Ŀ','fontsize',12)
ylim([-0.5 3.5])
