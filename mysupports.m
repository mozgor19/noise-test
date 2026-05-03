function [s1, s2, s3] = mysupports(f)
n = size(f);
for i = 1 : n(1) 
        s1(i) = sum(sum(f(i,:,:))) / ( n(2) * n(3) );
end
  s1 = ( s1 ) / sum( (1/n(1)) * s1.*s1 )^(1/2);


 for j = 1 : n(2) 
         s2(j) = sum(sum(f(:,j,:))) / ( n(1) * n(3) );
 end
 s2 = ( s2 ) / sum( (1/n(2)) * s2.*s2 )^(1/2); 


 for k = 1 : n(3) 
         s3(k) = sum(sum(f(:,:,k))) / ( n(1) * n(2) );
 end  
  s3 = ( s3 ) / sum( (1/n(3)) * s3.*s3 )^(1/2);
end