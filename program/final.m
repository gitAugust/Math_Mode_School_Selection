clear
clc
I=imread('t300.jpeg');%path�����·��
A=imread('1.jpeg');

I_R=I(:,:,1);%R����
I_G=I(:,:,2);%G����
I_B=I(:,:,3);%B����

A_R=A(:,:,1);%R����
A_G=A(:,:,2);%G����
A_B=A(:,:,3);%B����

[a,b,c,d]=mdwt(I_R-I_G);

m=nrss(a)

