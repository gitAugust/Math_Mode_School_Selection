clear ;clc;
for ji=3001:5888
    ji
    ii=int2str(ji);
    I=imread(['f',ii,'.jpg']);
    B=future(I);
    fp=fopen('F.txt','a');
    [m,n]=size(B);
    for i=1:1:m
        for j=1:1:n
            if j==n
                fprintf(fp,'%g\r\n',B(i,j));
            else
                fprintf(fp,'%g ',B(i,j));
            end
        end
    end
end