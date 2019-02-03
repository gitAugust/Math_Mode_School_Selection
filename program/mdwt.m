function [LL,HH] = mdwt(A)
%mdwt 一级二维小波变换函数 
%  输入：矩阵
%  输出：LL 对角线低频子带
%        HL 垂直高频子带
%        LH 水平高频子带
%        HH 对角线高频子带
%
[l,h]=size(A);
lhmin=min(l,h);
if mod(lhmin,2)==0
% a是偶数
else
% a是奇数
lhmin=lhmin-1;
end
f=A(1:lhmin,1:lhmin);%把矩阵变换为长宽一致切为偶数的矩阵
d=size(f);
T=d(1);
SUB_T=T/2;
%  2.进行二维小波分解
l=wfilters('db4','l');    %  db10（消失矩为10)低通分解滤波器冲击响应（长度为20）
L=T-length(l);
l_zeros=[l,zeros(1,L)];    %  矩阵行数与输入图像一致，为2的整数幂
h=wfilters('db4','h');    %  db10（消失矩为10)高通分解滤波器冲击响应（长度为20）
h_zeros=[h,zeros(1,L)];    %  矩阵行数与输入图像一致，为2的整数幂
for i=1:T   %  列变换
    row(1:SUB_T,i)=dyaddown( ifft( fft(l_zeros).*fft(f(:,i)') ) ).';    %  圆周卷积<->FFT
    row(SUB_T+1:T,i)=dyaddown( ifft( fft(h_zeros).*fft(f(:,i)') ) ).';  %  圆周卷积<->FFT
end
for j=1:T   %  行变换
    line(j,1:SUB_T)=dyaddown( ifft( fft(l_zeros).*fft(row(j,:)) ) );    %  圆周卷积<->FFT
    line(j,SUB_T+1:T)=dyaddown( ifft( fft(h_zeros).*fft(row(j,:)) ) );  %  圆周卷积<->FFT
end
decompose_pic=line;  %  分解矩阵
%  图像分为四块
lt_pic=decompose_pic(1:SUB_T,1:SUB_T);      %  在矩阵左上方为低频分量--fi(x)*fi(y)
rt_pic=decompose_pic(1:SUB_T,SUB_T+1:T);    %  矩阵右上为--fi(x)*psi(y)
lb_pic=decompose_pic(SUB_T+1:T,1:SUB_T);    %  矩阵左下为--psi(x)*fi(y)
rb_pic=decompose_pic(SUB_T+1:T,SUB_T+1:T);  %  右下方为高频分量--psi(x)*psi(y)
 
% colormap(map);
LL=abs(lt_pic);
% subplot(2,2,1);
% imshow(abs(lt_pic),[]);  %  左上方为低频分量--fi(x)*fi(y)
% title('\Phi(x)*\Phi(y)');
LH=rt_pic;
%subplot(2,2,2);
% imshow(abs(rt_pic),[]);  %  矩阵右上为--fi(x)*psi(y)
% title('\Phi(x)*\Psi(y)');
HL=lb_pic;
% subplot(2,2,3);
% imshow(abs(lb_pic),[]);  %  矩阵左下为--psi(x)*fi(y)
% title('\Psi(x)*\Phi(y)');
HH=abs(rb_pic);
% subplot(2,2,4);
% imshow(abs(rb_pic),[]);  %  右下方为高频分量--psi(x)*psi(y)
% title('\Psi(x)*\Psi(y)');
 
%  5.重构源图像及结果显示
% construct_pic=decompose_matrix'*decompose_pic*decompose_matrix;
% l_re=l_zeros(end:-1:1);   %  重构低通滤波
% l_r=circshift(l_re',1)';  %  位置调整
% h_re=h_zeros(end:-1:1);   %  重构高通滤波
% h_r=circshift(h_re',1)';  %  位置调整
% top_pic=[lt_pic,rt_pic];  %  图像上半部分
% t=0;
% for i=1:T  %  行插值低频
%  
%     if (mod(i,2)==0)
%         topll(i,:)=top_pic(t,:); %  偶数行保持
%     else
%         t=t+1;
%         topll(i,:)=zeros(1,T);   %  奇数行为零
%     end
% end
% for i=1:T  %  列变换
%     topcl_re(:,i)=ifft( fft(l_r).*fft(topll(:,i)') )';  %  圆周卷积<->FFT
% end
%  
% bottom_pic=[lb_pic,rb_pic];  %  图像下半部分
% t=0;
% for i=1:T  %  行插值高频
%     if (mod(i,2)==0)
%         bottomlh(i,:)=bottom_pic(t,:);  %  偶数行保持
%     else
%         bottomlh(i,:)=zeros(1,T);       %  奇数行为零
%         t=t+1;
%     end 
% end
% end

