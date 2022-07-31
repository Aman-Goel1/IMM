

clearvars
%addPath;
addpath('./motion')  
str0= './a08_s07s03_e01';
str3= ['2A_SBUIMM1sh_3']; %15 maybe 27
str4=['2B_SBUIMMsh_3'];  %15 maybe 27

%str=[str0,'_A.txt'];%%%%%%%%%%%%%%%
%str1=[str0,'_B.txt'];%read original B
%str1=[str0,'_fakeB_epoch2000.txt'];%read fake B
str=[str3];
str1=[str4];
% joint connectivity
J=[1 2 4 5 2 7 8 2 3 10 11 3 13 14; 
   2 4 5 6 7 8 9 3 10 11 12 13 14 15];

data_A=importdata(str);
data_B=importdata(str1);

num_frames=size(data_A,1);
data_B=data_B(1:num_frames,:);


data_A=reshape(data_A,num_frames,3,15);
X_A=data_A(:,1,:);
X_A=squeeze(X_A);
X_A=-X_A';


Y_A=data_A(:,2,:);
Y_A=squeeze(Y_A);
Y_A=2-Y_A';
%Y_A=Y_A';

Z_A=data_A(:,3,:);
Z_A=squeeze(Z_A);
Z_A=Z_A';


data_B=reshape(data_B,num_frames,3,15);
X_B=data_B(:,1,:);
X_B=squeeze(X_B);
X_B=-X_B';

Y_B=data_B(:,2,:);
Y_B=squeeze(Y_B);
Y_B=2-Y_B';
%Y_B=Y_B';

Z_B=data_B(:,3,:);
Z_B=squeeze(Z_B);
Z_B=Z_B';



for s=1:num_frames
    clf; % together with tubemesh
    S_A=[X_A(:,s) Z_A(:,s) Y_A(:,s)];
    S_B=[X_B(:,s) Z_B(:,s) Y_B(:,s)];

    xlim = [0 1];
    ylim = [0 1];
    zlim = [0 3];
    set(gca, 'xlim', xlim, ...
             'ylim', ylim, ...
             'zlim', zlim);

    plot3(S_A(:,1),S_A(:,2),S_A(:,3),'k.');
    hold on;
    plot3(S_B(:,1),S_B(:,2),S_B(:,3),'k.');
    hold off;
    set(gca,'DataAspectRatio',[1 1 1])
    %view([10,10,-90])
    view([74, 27]) % video view
    view([87,22])
    %view([73,90]) % top view
    %view([180,39]) % side view
    axis([-1 1 -0.5 1 1 2.5])%x,z,y  1 2.5
    xlabel('x');
    ylabel('z');
    zlabel('y');

    %text(0,0.5,1,['frame= ',num2str(s)])

    for j=1:size(J,2)
        c1=J(1,j);
        c2=J(2,j);
        line([S_A(c1,1) S_A(c2,1)], [S_A(c1,2) S_A(c2,2)],[S_A(c1,3) S_A(c2,3)], 'color','r','LineWidth',2);
        line([S_B(c1,1) S_B(c2,1)], [S_B(c1,2) S_B(c2,2)],[S_B(c1,3) S_B(c2,3)], 'color','b','LineWidth',2);
        %tubemesh([[S_A(c1,1);S_A(c2,1)], [S_A(c1,2);S_A(c2,2)],[S_A(c1,3);S_A(c2,3)]],0.03, [], @surf);
        %tubemesh([[S_B(c1,1);S_B(c2,1)], [S_B(c1,2);S_B(c2,2)],[S_B(c1,3);S_B(c2,3)]],0.03); view([85,-10]);%([10,50])
    end

    pause(1/7)    
end


%removePath;
