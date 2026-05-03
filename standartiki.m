function [ss,mean,b] = standartiki(a)


[n1,n2] = size(a);

for i=1:n1
    for j=1:n2
        if a(i,j)==255
           a(i,j)=0;
        end
    end
end

m=0;
k=0;

for i=1:n1
    for j=1:n2
        if a(i,j)~= 0
            m = m+a(i,j);
            k = k+1;
        end
    end
end

mean = m/k;
ss =0;
for i2=1:n1
    for j2=1:n2
        if a(i2,j2)~= 0
        ss = ss+ abs(mean-a(i2,j2));
        end
    end
end
ss = ss/k;

for i3=1:n1
    for j3=1:n2
        if a(i3,j3)==0
            a(i3,j3)=mean;
        end
    end
end




b=a;
end
    
    