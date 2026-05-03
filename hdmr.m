function [f0,f11,f12,f13] = hdmr(A1)
%[f0,f11,f12,f13,f2_12,f2_13,f2_23,B,C,D] = hdmr(A1)
A(:,:,1)=A1;
A(:,:,2)=A1;
A(:,:,3)=A1;
pic1=double(A);


dimensions = size(pic1);

for i = 1 : 3
  d(i)=dimensions(i);
  alpha(i) = 1/d(i);
end

%------------------------------------------------------------------
% Constant HDMR Term
%------------------------------------------------------------------
 f0=0;

  for i = 1 : d(1)
    for j = 1 : d(2)
      for k = 1 : d(3)
       f0=f0+alpha(1)*alpha(2)*alpha(3)*pic1(i,j,k);
      end
    end
  end
  
%------------------------------------------------------------------
% Univarite HDMR Term
%------------------------------------------------------------------

   for i = 1 : d(1)
     f11(i)=0;
     for j = 1 : d(2)
       for k = 1 : d(3)
         f11(i)= f11(i)+alpha(2)*alpha(3)*pic1(i,j,k);
       end
     end
      f11(i)= f11(i)-f0;
   end  
   
   for j = 1 : d(2)
     f12(j)=0;
     for i = 1 : d(1)  
       for k = 1 : d(3)
         f12(j)= f12(j)+alpha(1)*alpha(3)*pic1(i,j,k);
       end
       
     end
     f12(j)= f12(j)-f0;
   end

  for k = 1 : d(3)
     f13(k)=0;
     for i = 1 : d(1)  
       for j = 1 : d(2)
         f13(k)= f13(k)+alpha(1)*alpha(2)*pic1(i,j,k);
       end
       
     end
     f13(k)= f13(k)-f0;
   end

%------------------------------------------------------------------
% Bivariate HDMR Term
%------------------------------------------------------------------

 %  for i = 1 : d(1)
  %   for j = 1 : d(2)
   %     f2_12(i,j)=0;
    %    for k = 1 : d(3)
     %     f2_12(i,j)= f2_12(i,j)+alpha(3)*pic1(i,j,k);
     %   end
     %   f2_12(i,j)= f2_12(i,j)-f0-f11(i)-f12(j);   
     %end
   %end
   
   %for i = 1 : d(1)
   %  for k = 1 : d(3) 
   %     f2_13(i,k)=0;
   %    for j = 1 : d(2)
   %      f2_13(i,k)= f2_13(i,k)+alpha(2)*pic1(i,j,k);
   %    end
   %     f2_13(i,k)= f2_13(i,k)-f0-f11(i)-f13(k);   
   %  end
   %end

  %for j = 1 : d(2)
  %     for k = 1 : d(3)
  %          f2_23(j,k)=0;
  %        for i = 1 : d(1)  
  %         f2_23(j,k)= f2_23(j,k)+alpha(1)*pic1(i,j,k);
  %        end
  %        f2_23(j,k)= f2_23(j,k)-f0-f12(j)-f13(k); 
  %     end
  %end
   %----sonradan ekledim
   %B = zeros(d(1),d(2),d(3));
   %for i=1:d(1)
   %    for j=1:d(2)
   %        for k=1:d(3)
   %            B(i,j,k)=f11(i)+f12(j)+f13(k)+f0;
               
   %        end
   %    end
   % end
   %-------------------
   % C = zeros(d(1),d(2),d(3));
   %for i=1:d(1)
   %    for j=1:d(2)
   %        for k=1:d(3)
               %C(i,j,k)= f11(i)+f12(j)+f13(k);%+f2_23(j,k);
               % C(i,j,k)= f2_12(i,j)+f13(k)+f2_13(i,k)+f2_23(j,k);
    %             C(i,j,k)= f2_12(i,j)+f2_13(i,k)+f2_23(j,k)+0.9*f0;
     %      end
     %  end
   %end
       
   %-------------------
   % D = zeros(d(1),d(2),d(3));
   %for i=1:d(1)
   %    for j=1:d(2)
   %        for k=1:d(3)
   %            D(i,j,k)= f11(i)+f12(j)+f13(k);
               %D(i,j,k)= f11(i)+f12(j)+f13(k)+f0%+f2_12(i,j)%+f2_13(i,k)%+f2_23(j,k);
    %       end
    %   end
   %end
       
   %-------------------
end

