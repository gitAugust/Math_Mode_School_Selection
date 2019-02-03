function [S] = future(A)
%future ��ȡ�������ɫ����
%   �˴���ʾ��ϸ˵��
%1.��ȡrgbͨ��
R=A(:,:,1);%R����
G=A(:,:,2);%G����
B=A(:,:,3);%B����

%2.��RG��BG���в��
RG=R-G;
BG=B-G;
RB=R-B;

%3.��R,G,B,RG,BG�����˲�
[rll,rhh]=mdwt(R);
[gll,ghh]=mdwt(G);
[bll,bhh]=mdwt(B);
[rgll,rghh]=mdwt(RG);
[bgll,bghh]=mdwt(BG);
[rbll,rbhh]=mdwt(RB);


%4.����������
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

