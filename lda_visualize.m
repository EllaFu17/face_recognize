clear
clc
% load data
%   train_data  (1024,2982)
%   train_label (1,2982)
%   test_data   (1024,1278)
%   test_label  (1,1278)
load('../facedata.mat');

%% LDA & SVD
idx499 = randperm(2975,498);
idx1 = 2975 + randperm(3,2);
train_500 = [train_data(:,idx499) train_data(:,idx1)];
label_500 = [train_label(:,idx499) train_label(:,idx1)]; % 1*500

label_unique = unique(label_500);
avg = mean(train_500,2);
% cell
avg_i = cell(1,26);
train = cell(1,26);
s_i = cell(1,26);
num = cell(1,26);
sw = zeros(1024,1024);
sb = zeros(1024,1024);

for i = 1:26
    ind = find(label_500 == label_unique(i));
    train{i} = train_500(:,ind);
    avg_i{i} = mean(train{i},2);
    num{i} = size(train{i}, 2);
    s_i{i} = (train{i}-avg_i{i})*(train{i}-avg_i{i})'/num{i};
    sw = sw + s_i{i}*num{i}/500;
	sb = sb + (avg_i{i}-avg)*(avg_i{i}-avg)'*num{i}/500;
end

% [W,Lam] = eig(sb,sw);
% all_eigen_values = sum(Lam, 1);
% [~, I] = sort(all_eigen_values, 'descend');
% W = W(:, I); % 1024*1024

% svd
[U,D,V] = svd(pinv(sw)*sb);
W = U;
%% 2d
train_2d = W(:,1:2)' * train_data;% 40*2982,每一行是一个样本
figure()
hold on
grid on
s2d_pie = scatter(train_2d(1,idx499),train_2d(2,idx499), 20, train_label(1,idx499),'filled');
s2d_self = scatter(train_2d(1,idx1),train_2d(2,idx1), 72,'r','pentagram','filled');
title('LDA 2D ')
legend([s2d_pie s2d_self],{'PIE','SELF'})
hold off

%% 3d
train_3d = W(:,1:3)' * train_data;% 40*2982,每一行是一个样本
figure()
hold on
grid on
s3d_pie = scatter3(train_3d(1,idx499), train_3d(2,idx499), train_3d(3,idx499),20, train_label(1,idx499),'filled');
s3d_self = scatter3(train_3d(1,idx1), train_3d(2,idx1),train_3d(2,idx1), 72,'r','pentagram','filled');
view([1 2 1])
title('LDA 3D ')
legend([s3d_pie s3d_self],{'PIE','SELF'})
hold off