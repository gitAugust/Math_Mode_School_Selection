clear
clc
I=imread('t300.jpeg');%path是你的路径
A=imread('1.jpeg');

I_R=I(:,:,1);%R分量
I_G=I(:,:,2);%G分量
I_B=I(:,:,3);%B分量

A_R=A(:,:,1);%R分量
A_G=A(:,:,2);%G分量
A_B=A(:,:,3);%B分量

[a,b,c,d]=mdwt(I_R-I_G);

m=nrss(a)

