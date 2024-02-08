% Script to run MDH based NRSfM on the KINECT Paper sequence
clear all;
close all;
addpath(genpath('../../libs/yalmip/'));
addpath('/home/ajad/PhD/libs/mosek/7/toolbox/r2013aom/');
load KinectPaper; % obtained from Varol et al. data.
load camcalib;
sp = 5; % subsample points
totframes = length(p); % number of images
for k = 1: totframes    
    % get normalized points from calibration
    m(k).m = inv(KK)* [p(k).p(1:2,1:sp:end); ones(1,length(p(k).p(:,1:sp:end)))];
    m(k).m = m(k).m(1:2,:);
    Pgth(k).P = Pgth(k).P(:,1:sp:end);            
end

sv = 8; % subsample views
Pgt = Pgth(sv:sv:end);
m = m(sv:sv:end);
N = length(m(1).m);
M = length(m);
% visibility is true for the example:
visibt = true(N,M);
Kneighbors = 20;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
IDX = getNeighborsVis(m,Kneighbors,visibt);
visbc = num2cell(visibt,1);

% second part: formulate socp and solve
mc  = squeeze(struct2cell(m));
disp('NRSfM function');
tic;
[mu,D] = NrSfM(IDX,mc,visbc);
ts = toc;

% third part: display results, 
%%
res.Q2 = cell(1,M);
res.Pg = cell(1,M);
res.err3d = zeros(1,M); % RMSE for each surface
res.err3dper = zeros(1,M); % RMSE for each surface
for k=1:M
    Q2k=double([mu(k,visibt(:,k));mu(k,visibt(:,k));mu(k,visibt(:,k))]).*[m(k).m(:,visibt(:,k));ones(1,length(m(k).m(:,visibt(:,k))))];    
    P2 = Pgt(k).P(:,visibt(:,k));
    % get valid indices: some groundtruth points are 0
    mugth = P2(3,:);
    l = mugth>0;
    % fix scale of reconstructed surface
    Q2k_n = RegisterToGTH(Q2k(:,l),P2(:,l));
    
    figure(1)
    clf;
    plot3(Q2k_n(1,:),Q2k_n(2,:),Q2k_n(3,:),'b*');
    hold on;
    plot3(P2(1,l),P2(2,l),P2(3,l),'go');
    hold off;
    pause(0.2);
    res.Q2{k} = Q2k_n;
    res.Pg{k} = P2(:,l);
 
    scale = norm(P2(:,l),'fro');    
    res.err3dper(k) = norm(Q2k_n - P2(:,l),'fro')/scale*100;    
    res.err3d(k) = sqrt(mean(sum((P2(:,l)-Q2k_n).^2)));
    fprintf('3D rmse =%.2f mm\t',res.err3d(k));
    fprintf('relative 3D error =%.2f %% \n',res.err3dper(k));
    pause(0.2);        
end
meandepth = mean(res.err3d)
meanper = mean(res.err3dper)

save('results-KPaper','res','ts');