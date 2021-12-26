clear
clc
load('../facedata.mat');
%   train_data  (1024,2982)
%   train_label (1,2982)
%   test_data   (1024,1278)
%   test_label  (1,1278)
[train_m, train_n] = size(train_data);
%% PCA & SVD
idx499 = randperm(2975,498);
idx1 = 2975 + randperm(3,2);
train_500 = [train_data(:,idx499) train_data(:,idx1)];
% centralize the data
train_500mean = mean(train_500,2); % average by row，1024*1
train_500cen = train_500 - train_500mean;
% svd
[U,D,V] = svd(train_500cen);
lambda = D*D';
% k-nearest neighbor
k = 1;
%% 3 eigenfaces
face1 = reshape(U(:,1),32,32);
face2 = reshape(U(:,2),32,32);
face3 = reshape(U(:,3),32,32);
% face1 = (face1 - min(face1,[],'all'))./max(face1,[],'all');
% face2 = (face2 - min(face2,[],'all'))./max(face2,[],'all');
% face3 = (face3 - min(face3,[],'all'))./max(face3,[],'all');
figure()
hold on
subplot(1,3,1);imshow(face1, []);
title(sprintf('eigenface1'));
subplot(1,3,2);imshow(face2, []);
title(sprintf('eigenface2'));
subplot(1,3,3);imshow(face3, []);
title(sprintf('eigenface3'));
hold off
%% 2d
train_2d = U(:,1:2)' * train_data;% 40*2982,每一行是一个样本
figure()
hold on
grid on
s2d_pie = scatter(train_2d(1,idx499),train_2d(2,idx499), 20, train_label(1,idx499),'filled');
s2d_self = scatter(train_2d(1,idx1),train_2d(2,idx1), 72,'r','pentagram','filled');
title('PCA 2D ')
legend([s2d_pie s2d_self],{'PIE','SELF'})
hold off

%% 3d
train_3d = U(:,1:3)' * train_data;% 40*2982,每一行是一个样本
figure()
hold on
grid on
s3d_pie = scatter3(train_3d(1,idx499), train_3d(2,idx499), train_3d(3,idx499),20, train_label(1,idx499),'filled');
s3d_self = scatter3(train_3d(1,idx1), train_3d(2,idx1),train_3d(2,idx1), 72,'r','pentagram','filled');
view([1 2 1])
title('PCA 3D ')
legend([s3d_pie s3d_self],{'PIE','SELF'})
hold off

