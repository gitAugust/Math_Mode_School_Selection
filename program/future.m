function [S] = future(A)
%future 提取矩阵的颜色特征
%   此处显示详细说明
%1.提取rgb通道
R=A(:,:,1);%R分量
G=A(:,:,2);%G分量
B=A(:,:,3);%B分量

%2.对RG、BG进行差分
RG=R-G;
BG=B-G;
RB=R-B;

%3.对R,G,B,RG,BG进行滤波
[rll,rhh]=mdwt(R);
[gll,ghh]=mdwt(G);
[bll,bhh]=mdwt(B);
[rgll,rghh]=mdwt(RG);
[bgll,bghh]=mdwt(BG);
[rbll,rbhh]=mdwt(RB);


%4.计算清晰度
S(1)=nrss(rll);
S(2)=nrss(gll);
S(3)=nrss(bll);
S(4)=nrss(rghh);
S(5)=nrss(rgll);
S(6)=nrss(bghh);
S(7)=nrss(bgll);
S(8)=nrss(rbhh);
S(9)=nrss(rbll);
end

