function a = nrss(C)
%UNTITLED8 此处显示有关此函数的摘要
%%计算图像清晰度NRSS
%%reference paper:一种针对图像模糊的无参考质量评价指标，计算机应用，谢小甫等。
img = C;  %读取原始图像
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
[m,n] = size(img);
gausFilter = fspecial('gaussian',[7 7],sigma);  %构建高斯滤波器
Ir = imfilter(img,gausFilter,'replicate');  %高斯滤波
%% (2)利用Sobel算子计算图像img和Ir的梯度图像G和Gr
G = edge(img,'sobel');  %用Sobel算子求梯度图像
Gr= edge(Ir,'sobel');
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
a=nrss;

end

