function [Xyak, f0] = EMPR0(s1,s2,s3,Xem)
       n = size(Xem);
       
       f0=0;
                for i = 1:n(1)
                    for j = 1:n(2)
                        for k = 1:n(3)
                        f0=f0+(Xem(i,j,k)*s1(i)*s2(j)*s3(k));
                       end
                    end
                end
                f0=f0*(1/(n(1)*n(2)*n(3)));
                for i = 1 : n(1) % Dis Carpim
                    for j = 1 : n(2) 
                        for k = 1 : n(3)
                            Xyak(i,j,k) = f0*s1(i)*s2(j)*s3(k);
                        end
                    end
                end
       
end
  
  
 
  




