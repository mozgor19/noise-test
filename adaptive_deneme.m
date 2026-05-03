clear all
close all
%%% YBMG ile uyarlamalı pencereli filtre (son)
I=imread('barbara512.tif');
I = I(:,:,1);
imshow(I);
figure;
I2=imnoise(I,'salt & pepper',0.2);
imshow(I2);
imshow(I2);
A=double(I2);
[m,n]=size(A);
for i=1:m
    for j=1:n
        if A(i,j)==255
           A(i,j)=0;
        end
    end
end

pA=padarray(A,[3 3],'symmetric');
[m,n]=size(A);

for i=1:m
                for j=1:n  
                    if(A(i,j)==0)                       
                          if(sum(sum(pA(i+2:i+4,j+2:j+4)))~=0)                              
                            R1=pA(i+2:i+4,j+2:j+4);
                            [ss,mean,b] = standartiki(R1); 
                            [f0,f11,f12,f13] = hdmr(b);
                            A(i,j)=(median(f11)+median(f12))/2+f0;
                            
                                                          
                          elseif(sum(sum(pA(i+1:i+5,j+1:j+5)))~=0)
                            R1=pA(i+1:i+5,j+1:j+5);
                            [ss,mean,b] = standartiki(R1); 
                            [f0,f11,f12,f13] = hdmr(b);
                            A(i,j)=(median(f11)+median(f12))/2+f0;
                           
                            
                          elseif(sum(sum(pA(i:i+6,j:j+6)))~=0)                              
                           R1=pA(i:i+6,j:j+6);
                           [ss,mean,b] = standartiki(R1); 
                           [f0,f11,f12,f13] = hdmr(b);
                           A(i,j)=(median(f11)+median(f12))/2+f0;
                          
                            
                          end                   
                    end
                end    
            end

     for kk=1:3       
pA=padarray(A,[1 1],'symmetric');
[m,n]=size(A);
  
          for i=1:m
                for j=1:n  
                    if(A(i,j)==0)                       
                          if(sum(sum(pA(i:i+2,j:j+2)))~=0)                              
                            R1=pA(i:i+2,j:j+2);
                            [ss,mean,b] = standartiki(R1); 
                            [f0,f11,f12,f13] = hdmr(b);
                            A(i,j)=(median(f11)+median(f12))/2+f0;
                          
                          end
                    end
                end
          end
     end

            A=uint8(A);
            figure;
            imshow(A);

%H=zeros(294,842);

%for i=1:294
%    for j=1:842
%       if H(i,j)==0
%           H(i,j)=255;
%       end
%    end
% end
%H(20:275,20:275)=double(I);
%H(20:275,294:549)=double(I2);
%H(20:275,568:823)=double(A);
%imshow(uint8(H));

            psnrhdmr=psnr(I,A)
            ssimhdmr=ssim(I,A)
           %fsim_value = kenar(I, A)
           ms_ssim_value = multissim(I, A)
 
            




