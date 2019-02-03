function [LL,HH] = mdwt(A)
%mdwt һ����άС���任���� 
%  ���룺����
%  �����LL �Խ��ߵ�Ƶ�Ӵ�
%        HL ��ֱ��Ƶ�Ӵ�
%        LH ˮƽ��Ƶ�Ӵ�
%        HH �Խ��߸�Ƶ�Ӵ�
%
[l,h]=size(A);
lhmin=min(l,h);
if mod(lhmin,2)==0
% a��ż��
else
% a������
lhmin=lhmin-1;
end
f=A(1:lhmin,1:lhmin);%�Ѿ���任Ϊ����һ����Ϊż���ľ���
d=size(f);
T=d(1);
SUB_T=T/2;
%  2.���ж�άС���ֽ�
l=wfilters('db4','l');    %  db10����ʧ��Ϊ10)��ͨ�ֽ��˲��������Ӧ������Ϊ20��
L=T-length(l);
l_zeros=[l,zeros(1,L)];    %  ��������������ͼ��һ�£�Ϊ2��������
h=wfilters('db4','h');    %  db10����ʧ��Ϊ10)��ͨ�ֽ��˲��������Ӧ������Ϊ20��
h_zeros=[h,zeros(1,L)];    %  ��������������ͼ��һ�£�Ϊ2��������
for i=1:T   %  �б任
    row(1:SUB_T,i)=dyaddown( ifft( fft(l_zeros).*fft(f(:,i)') ) ).';    %  Բ�ܾ��<->FFT
    row(SUB_T+1:T,i)=dyaddown( ifft( fft(h_zeros).*fft(f(:,i)') ) ).';  %  Բ�ܾ��<->FFT
end
for j=1:T   %  �б任
    line(j,1:SUB_T)=dyaddown( ifft( fft(l_zeros).*fft(row(j,:)) ) );    %  Բ�ܾ��<->FFT
    line(j,SUB_T+1:T)=dyaddown( ifft( fft(h_zeros).*fft(row(j,:)) ) );  %  Բ�ܾ��<->FFT
end
decompose_pic=line;  %  �ֽ����
%  ͼ���Ϊ�Ŀ�
lt_pic=decompose_pic(1:SUB_T,1:SUB_T);      %  �ھ������Ϸ�Ϊ��Ƶ����--fi(x)*fi(y)
rt_pic=decompose_pic(1:SUB_T,SUB_T+1:T);    %  ��������Ϊ--fi(x)*psi(y)
lb_pic=decompose_pic(SUB_T+1:T,1:SUB_T);    %  ��������Ϊ--psi(x)*fi(y)
rb_pic=decompose_pic(SUB_T+1:T,SUB_T+1:T);  %  ���·�Ϊ��Ƶ����--psi(x)*psi(y)
 
% colormap(map);
LL=abs(lt_pic);
% subplot(2,2,1);
% imshow(abs(lt_pic),[]);  %  ���Ϸ�Ϊ��Ƶ����--fi(x)*fi(y)
% title('\Phi(x)*\Phi(y)');
LH=rt_pic;
%subplot(2,2,2);
% imshow(abs(rt_pic),[]);  %  ��������Ϊ--fi(x)*psi(y)
% title('\Phi(x)*\Psi(y)');
HL=lb_pic;
% subplot(2,2,3);
% imshow(abs(lb_pic),[]);  %  ��������Ϊ--psi(x)*fi(y)
% title('\Psi(x)*\Phi(y)');
HH=abs(rb_pic);
% subplot(2,2,4);
% imshow(abs(rb_pic),[]);  %  ���·�Ϊ��Ƶ����--psi(x)*psi(y)
% title('\Psi(x)*\Psi(y)');
 
%  5.�ع�Դͼ�񼰽����ʾ
% construct_pic=decompose_matrix'*decompose_pic*decompose_matrix;
% l_re=l_zeros(end:-1:1);   %  �ع���ͨ�˲�
% l_r=circshift(l_re',1)';  %  λ�õ���
% h_re=h_zeros(end:-1:1);   %  �ع���ͨ�˲�
% h_r=circshift(h_re',1)';  %  λ�õ���
% top_pic=[lt_pic,rt_pic];  %  ͼ���ϰ벿��
% t=0;
% for i=1:T  %  �в�ֵ��Ƶ
%  
%     if (mod(i,2)==0)
%         topll(i,:)=top_pic(t,:); %  ż���б���
%     else
%         t=t+1;
%         topll(i,:)=zeros(1,T);   %  ������Ϊ��
%     end
% end
% for i=1:T  %  �б任
%     topcl_re(:,i)=ifft( fft(l_r).*fft(topll(:,i)') )';  %  Բ�ܾ��<->FFT
% end
%  
% bottom_pic=[lb_pic,rb_pic];  %  ͼ���°벿��
% t=0;
% for i=1:T  %  �в�ֵ��Ƶ
%     if (mod(i,2)==0)
%         bottomlh(i,:)=bottom_pic(t,:);  %  ż���б���
%     else
%         bottomlh(i,:)=zeros(1,T);       %  ������Ϊ��
%         t=t+1;
%     end 
% end
% end

