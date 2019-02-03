clear
clc
I=imread('5.jpeg');%path是你的路径
A=imread('1.jpeg');

I_R=I(:,:,1);%R分量
I_G=I(:,:,2);%G分量
I_B=I(:,:,3);%B分量

A_R=A(:,:,1);%R分量
A_G=A(:,:,2);%G分量
A_B=A(:,:,3);%B分量

[a,d]=mdwt(I_R-I_G);

%%计算图像清晰度NRSS
%%reference paper:一种针对图像模糊的无参考质量评价指标，计算机应用，谢小甫等。

img = a;  %读取原始图像
N = 64;  %取方差最大的前N块
block_size = 8;  %块大小
stride = 4;  %分块步长（小于block_size时有重叠部分）
 
%%初始化
blk_count = 0;
ssim_sum = 0;
G_blk = zeros(block_size,block_size,1);
Gr_blk = zeros(block_size,block_size,1);
G_std = zeros(1);
 
%% (1)低通滤波,生成参考图像Ir
sigma = sqrt(6);
if size(img,3) == 3
    img = rgb2gray(img);  %灰度化
end
[m,n] = size(img);
gausFilter = fspecial('gaussian',[7 7],sigma);  %构建高斯滤波器
Ir = imfilter(img,gausFilter,'replicate');  %高斯滤波
figure,
subplot(121),imshow(img),title('Original Image');
subplot(122),imshow(Ir),title('Gaussian Filter Image');
 
%% (2)利用Sobel算子计算图像img和Ir的梯度图像G和Gr
G = edge(img,'sobel');  %用Sobel算子求梯度图像
Gr= edge(Ir,'sobel');
figure,
subplot(121),imshow(G),title('G');
subplot(122),imshow(Gr),title('Gr');
 
%% (3)将梯度图像划分成小块并计算每块的方差，找出其中方差最大的前N个
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
    G_std(i) = (std2(G_blk(:,:,i)))^2;  %计算方差
end
G_std_sort = sort(G_std,'descend');  %方差降序排列
G_indice = find(G_std >= G_std_sort(N));
G_indice = G_indice(1:N);  %取前N个的下标
 
%% (4)计算图像的无参考结构清晰度
for i = G_indice
    mssim = ssim(G_blk(:,:,i),Gr_blk(:,:,i));
    ssim_sum = ssim_sum+mssim;
end
nrss = 1-ssim_sum/N;%利用公式计算得到NRSS
display(nrss);
% eda = edge(a, 'canny', 0.5); 
% edb = edge(b, 'canny', 0.5); 
% edc = edge(c, 'canny', 0.5); 
% figure(3)    
%     subplot(4,2,1);
%         imshow(a);%原图
%     subplot(4,2,2);
%         imshow(eda);%高斯滤波后
%     subplot(4,2,3);
%         imshow(b);%导数
%     subplot(4,2,4);
%         imshow(edb);%非极大值抑制
%     subplot(4,2,5);
%         imshow(c);%双阈值
%     subplot(4,2,6);
%         imshow(edc);%Matlab自带边缘检测

% function [LL,HL,LH,HH] = mdwt(A)
% %UNTITLED2 此处显示有关此函数的摘要
% %   此处显示详细说明
% 
% f=A(1:374,1:374);%把矩阵变换为长宽一致切为偶数的矩阵
% d=size(f);
% if length(d)>2
%     f=rgb2gray((f));%%%%%%%%如果是彩色图像则转化为灰度图
% end
% T=d(1);
% SUB_T=T/2;
% %  2.进行二维小波分解
% l=wfilters('db10','l');    %  db10（消失矩为10)低通分解滤波器冲击响应（长度为20）
% L=T-length(l);
% l_zeros=[l,zeros(1,L)];    %  矩阵行数与输入图像一致，为2的整数幂
% h=wfilters('db10','h');    %  db10（消失矩为10)高通分解滤波器冲击响应（长度为20）
% h_zeros=[h,zeros(1,L)];    %  矩阵行数与输入图像一致，为2的整数幂
% for i=1:T   %  列变换
%     row(1:SUB_T,i)=dyaddown( ifft( fft(l_zeros).*fft(f(:,i)') ) ).';    %  圆周卷积<->FFT
%     row(SUB_T+1:T,i)=dyaddown( ifft( fft(h_zeros).*fft(f(:,i)') ) ).';  %  圆周卷积<->FFT
% end
% for j=1:T   %  行变换
%     line(j,1:SUB_T)=dyaddown( ifft( fft(l_zeros).*fft(row(j,:)) ) );    %  圆周卷积<->FFT
%     line(j,SUB_T+1:T)=dyaddown( ifft( fft(h_zeros).*fft(row(j,:)) ) );  %  圆周卷积<->FFT
% end
% decompose_pic=line;  %  分解矩阵
% %  图像分为四块
% lt_pic=decompose_pic(1:SUB_T,1:SUB_T);      %  在矩阵左上方为低频分量--fi(x)*fi(y)
% rt_pic=decompose_pic(1:SUB_T,SUB_T+1:T);    %  矩阵右上为--fi(x)*psi(y)
% lb_pic=decompose_pic(SUB_T+1:T,1:SUB_T);    %  矩阵左下为--psi(x)*fi(y)
% rb_pic=decompose_pic(SUB_T+1:T,SUB_T+1:T);  %  右下方为高频分量--psi(x)*psi(y)
%  
% %  3.分解结果显示
% figure(1);
% subplot(2,1,1);
% imshow(f,[]);  %  原始图像  
% title('original pic');
% subplot(2,1,2);
% image(abs(decompose_pic));  %  分解后图像
% title('decomposed pic');
% figure(2);
% % colormap(map);
% LL=lt_pic;
% subplot(2,2,1);
% imshow(abs(lt_pic),[]);  %  左上方为低频分量--fi(x)*fi(y)
% title('\Phi(x)*\Phi(y)');
% subplot(2,2,2);
% LH=rt_pic;
% imshow(abs(rt_pic),[]);  %  矩阵右上为--fi(x)*psi(y)
% title('\Phi(x)*\Psi(y)');
% HL=lb_pic;
% subplot(2,2,3);
% imshow(abs(lb_pic),[]);  %  矩阵左下为--psi(x)*fi(y)
% title('\Psi(x)*\Phi(y)');
% HH=rb_pic;
% subplot(2,2,4);
% imshow(abs(rb_pic),[]);  %  右下方为高频分量--psi(x)*psi(y)
% title('\Psi(x)*\Psi(y)');
%  
% %  5.重构源图像及结果显示
% % construct_pic=decompose_matrix'*decompose_pic*decompose_matrix;
% % l_re=l_zeros(end:-1:1);   %  重构低通滤波
% % l_r=circshift(l_re',1)';  %  位置调整
% % h_re=h_zeros(end:-1:1);   %  重构高通滤波
% % h_r=circshift(h_re',1)';  %  位置调整
% % top_pic=[lt_pic,rt_pic];  %  图像上半部分
% % t=0;
% % for i=1:T  %  行插值低频
% %  
% %     if (mod(i,2)==0)
% %         topll(i,:)=top_pic(t,:); %  偶数行保持
% %     else
% %         t=t+1;
% %         topll(i,:)=zeros(1,T);   %  奇数行为零
% %     end
% % end
% % for i=1:T  %  列变换
% %     topcl_re(:,i)=ifft( fft(l_r).*fft(topll(:,i)') )';  %  圆周卷积<->FFT
% % end
% %  
% % bottom_pic=[lb_pic,rb_pic];  %  图像下半部分
% % t=0;
% % for i=1:T  %  行插值高频
% %     if (mod(i,2)==0)
% %         bottomlh(i,:)=bottom_pic(t,:);  %  偶数行保持
% %     else
% %         bottomlh(i,:)=zeros(1,T);       %  奇数行为零
% %         t=t+1;
% %     end 
% % end
% % end



%%%%%%%%%%%%%%bp%%%%%%%%%%%%%%
clc, clear

tr=load('train_data.txt');
te=load('test_data.txt');

a=load('SMV.txt'); %把表中x1...x8的所有数据保存在纯文本文件fenlei.txt中
a=a(:,[1,3,5,7,8,9,10]);
%[2,3,5,7,8,9]
nu=500;
f=tr(1:nu,:);
t=tr(5001:(nu+5000),:);
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


%分类绘图
%制图_预测分类标签和实际分类标签对比图
figure(1)
plot(y,'og')
hold on
plot(label_test,'r*');
legend('预测标签','实际标签')
title('BP神经网络预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
ylim([-0.5 3.5])
