function[Xyak1, f1, f2, f3,m1] = EMPR1(f0,s1,s2,s3,Xem)

n = size(Xem);

for i = 1 : n(1) %1. yon icin bir yoney hesaplanacak
        f1(i) = 0;
        for j = 1 : n(2)
            for k = 1 : n(3)
                f1(i) = f1(i) + Xem(i,j,k) * s2(j) * s3(k) / (n(2) * n(3));                
            end
           
        end
        f1(i) = f1(i) - f0 * s1(i);
 end
    
    
  

  for j = 1 : n(2) %2. yon icin bir yoney hesaplanacak
        f2(j) = 0;
        for i = 1 : n(1)
            for k = 1 : n(3)
                f2(j) = f2(j) + Xem(i,j,k)*s1(i)*s3(k)/(n(1)*n(3));
                
            end
           
        end
        f2(j)=f2(j)-f0*s2(j); 
 end
        
    
    
    
  for k = 1 : n(3) %3. yon icin bir yoney hesaplanacak
        f3(k) = 0;
        for i = 1 : n(1)
            for j = 1 : n(2)
                f3(k) = f3(k) + Xem(i,j,k)*s1(i)*s2(j)/(n(1)*n(2));
                
            end
           
        end
        f3(k)=f3(k)-f0*s3(k);   
        
 end
        
    
    
  
   for i = 1 : n(1) % Dis Carpim
            for j = 1 : n(2) 
                for k = 1 : n(3)
                Xyak1(i,j,k) = f0*s1(i)*s2(j)*s3(k) + f1(i)*s2(j)*s3(k) + f1(i)*s2(j)*s3(k) + s1(i)*s2(j)*f3(k);
                m1=(median(median(f1(i)*s2(j)*s3(k)))+median(median(f1(i)*s2(j)*s3(k))))/2+f0*s1(i)*s2(j)*s3(k);
                %m2= (median(median(f1(i)*s2(j)*s3(k)))+median(median(f1(i)*s2(j)*s3(k))))/2;
                
                end
            end
   end
end