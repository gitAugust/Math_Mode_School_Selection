clc;




for ji=50:100
    ii=int2str(ji);
    A=imread(['t',ii,'.jpeg']);
    R=A(:,:,1);%R����
    G=A(:,:,2);%G����
    B=A(:,:,3);%B����
    RG=abs(R-G);
    BG=abs(B-G);
    RB=abs(R-B);
    rg= sum(RG(:));
    bg=sum(BG(:));
    rb=sum(RB(:));
    ttt=sqrt(rg^2+bg^2+rb^2);
    S(1,ji)=ttt;
    fp=fopen('rg rb bg.txt','a');
    fprintf(fp,'%g ',ttt);
    fclose(fp);
end

