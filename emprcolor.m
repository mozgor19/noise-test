clear all
close all

%I=imread('peppercolor.tif');
I=imread("C:\Users\Template\Desktop\noise-test\Tiny_Noisy_10\test_2.JPEG");
%I=imread('peppers.png');

B(:,:,1)=I(:,:,1);
B(:,:,2)=I(:,:,2);
B(:,:,3)=I(:,:,3);

imshow(B);
%A=imnoise(B,'salt & pepper',0.9);
A=B;
figure
imshow(A);



A=double(A);

A1=A(:,:,1);
A2=A(:,:,2);
A3=A(:,:,3);

[m,n]=size(A1);


%%%% A1 için%%%%%
for i=1:m
    for j=1:n
        if A1(i,j)==255
           A1(i,j)=0;
        end
    end
end

pA=padarray(A1,[3 3],'symmetric');
[m,n]=size(A1);

for i=1:m
                for j=1:n  
                    if(A1(i,j)==0)                       
                          if(sum(sum(pA(i+2:i+4,j+2:j+4)))~=0)                              
                            R1=pA(i+2:i+4,j+2:j+4);
                            [ss,mean,b] = standartiki(R1);
                            c(:,:,1)=b;
                            c(:,:,2)=b;
                            c(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c);
                            A1(i,j)=median(median(f1)+(median(f2)))+f0;


                            
                            
                                                          
                          elseif(sum(sum(pA(i+1:i+5,j+1:j+5)))~=0)
                            R1=pA(i+1:i+5,j+1:j+5);
                            [ss,mean,b] = standartiki(R1); 
                            c2(:,:,1)=b;
                            c2(:,:,2)=b;
                            c2(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c2);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c2);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c2);
                            A1(i,j)=median(median(f1)+(median(f2)))+f0;
                            
                          elseif(sum(sum(pA(i:i+6,j:j+6)))~=0)                              
                           R1=pA(i:i+6,j:j+6);
                           [ss,mean,b] = standartiki(R1); 
                           c3(:,:,1)=b;
                            c3(:,:,2)=b;
                            c3(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c3);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c3);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c3);
                            A1(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                            
                          end                   
                    end
                end    
            end

     for kk=1:3       
pA=padarray(A1,[1 1],'symmetric');
[m,n]=size(A1);
  
          for i=1:m
                for j=1:n  
                    if(A1(i,j)==0)                       
                          if(sum(sum(pA(i:i+2,j:j+2)))~=0)                              
                            R1=pA(i:i+2,j:j+2);
                            [ss,mean,b] = standartiki(R1); 
                            c4(:,:,1)=b;
                            c4(:,:,2)=b;
                            c4(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c4);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c4);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c4);
                            A1(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                          end
                    end
                end
          end
     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%55



%%%% A2 için  %%%%%%%%%%
  for i=1:m
    for j=1:n
        if A2(i,j)==255
           A2(i,j)=0;
        end
    end
end

pA=padarray(A2,[3 3],'symmetric');
[m,n]=size(A2);

for i=1:m
                for j=1:n  
                    if(A2(i,j)==0)                       
                          if(sum(sum(pA(i+2:i+4,j+2:j+4)))~=0)                              
                            R1=pA(i+2:i+4,j+2:j+4);
                            [ss,mean,b] = standartiki(R1); 
                            c(:,:,1)=b;
                            c(:,:,2)=b;
                            c(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c);
                            A2(i,j)=median(median(f1)+(median(f2)))+f0;
                            
                                                          
                          elseif(sum(sum(pA(i+1:i+5,j+1:j+5)))~=0)
                            R1=pA(i+1:i+5,j+1:j+5);
                            [ss,mean,b] = standartiki(R1); 
                            c2(:,:,1)=b;
                            c2(:,:,2)=b;
                            c2(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c2);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c2);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c2);
                            A2(i,j)=median(median(f1)+(median(f2)))+f0;
                           
                            
                          elseif(sum(sum(pA(i:i+6,j:j+6)))~=0)                              
                           R1=pA(i:i+6,j:j+6);
                           [ss,mean,b] = standartiki(R1); 
                           c3(:,:,1)=b;
                            c3(:,:,2)=b;
                            c3(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c3);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c3);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c3);
                            A2(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                            
                          end                   
                    end
                end    
            end

     for kk=1:3       
pA=padarray(A2,[1 1],'symmetric');
[m,n]=size(A2);
  
          for i=1:m
                for j=1:n  
                    if(A2(i,j)==0)                       
                          if(sum(sum(pA(i:i+2,j:j+2)))~=0)                              
                            R1=pA(i:i+2,j:j+2);
                            [ss,mean,b] = standartiki(R1); 
                            c4(:,:,1)=b;
                            c4(:,:,2)=b;
                            c4(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c4);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c4);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c4);
                            A2(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                          end
                    end
                end
          end
     end

%%%%%%%%%%%%%%%%%%

%%%% A3 için %%%%%
  for i=1:m
    for j=1:n
        if A3(i,j)==255
           A3(i,j)=0;
        end
    end
end

pA=padarray(A3,[3 3],'symmetric');
[m,n]=size(A3);

for i=1:m
                for j=1:n  
                    if(A3(i,j)==0)                       
                          if(sum(sum(pA(i+2:i+4,j+2:j+4)))~=0)                              
                            R1=pA(i+2:i+4,j+2:j+4);
                            [ss,mean,b] = standartiki(R1); 
                            c(:,:,1)=b;
                            c(:,:,2)=b;
                            c(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c);
                            A3(i,j)=median(median(f1)+(median(f2)))+f0;
                            
                                                          
                          elseif(sum(sum(pA(i+1:i+5,j+1:j+5)))~=0)
                            R1=pA(i+1:i+5,j+1:j+5);
                            [ss,mean,b] = standartiki(R1); 
                            c2(:,:,1)=b;
                            c2(:,:,2)=b;
                            c2(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c2);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c2);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c2);
                            A3(i,j)=median(median(f1)+(median(f2)))+f0;
                           
                            
                          elseif(sum(sum(pA(i:i+6,j:j+6)))~=0)                              
                           R1=pA(i:i+6,j:j+6);
                           [ss,mean,b] = standartiki(R1); 
                           c3(:,:,1)=b;
                            c3(:,:,2)=b;
                            c3(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c3);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c3);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c3);
                            A3(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                            
                          end                   
                    end
                end    
            end

     for kk=1:3       
pA=padarray(A3,[1 1],'symmetric');
[m,n]=size(A3);
  
          for i=1:m
                for j=1:n  
                    if(A3(i,j)==0)                       
                          if(sum(sum(pA(i:i+2,j:j+2)))~=0)                              
                            R1=pA(i:i+2,j:j+2);
                            [ss,mean,b] = standartiki(R1); 
                            c4(:,:,1)=b;
                            c4(:,:,2)=b;
                            c4(:,:,3)=b;
                            [s1, s2, s3] = mysupports(c4);
                            [Xyak, f0] = EMPR0(s1,s2,s3,c4);
                            [Xyak1, f1, f2, f3] = EMPR1(f0,s1,s2,s3,c4);
                            A3(i,j)=median(median(f1)+(median(f2)))+f0;
                          
                          end
                    end
                end
          end
     end
%%%%%%


AA(:,:,1)=A1;
AA(:,:,2)=A2;
AA(:,:,3)=A3;
            AA=uint8(AA);
            figure;
            imshow(AA);
            
            psnrempr=psnr(AA,B)
            ssimempr=ssim(AA,B)
            %dE = imcolordiff(AA,B);
            %imshow(dE);

