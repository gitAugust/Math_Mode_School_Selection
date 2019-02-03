function a = nrss(C)
%UNTITLED8 �˴���ʾ�йش˺�����ժҪ
%%����ͼ��������NRSS
%%reference paper:һ�����ͼ��ģ�����޲ο���������ָ�꣬�����Ӧ�ã�лС���ȡ�
img = C;  %��ȡԭʼͼ��
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
[m,n] = size(img);
gausFilter = fspecial('gaussian',[7 7],sigma);  %������˹�˲���
Ir = imfilter(img,gausFilter,'replicate');  %��˹�˲�
%% (2)����Sobel���Ӽ���ͼ��img��Ir���ݶ�ͼ��G��Gr
G = edge(img,'sobel');  %��Sobel�������ݶ�ͼ��
Gr= edge(Ir,'sobel');
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
a=nrss;

end

