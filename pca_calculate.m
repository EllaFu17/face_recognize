clear
clc
load('../facedata.mat');
%   train_data  (1024,2982)
%   train_label (1,2982)
%   test_data   (1024,1278)
%   test_label  (1,1278)
[train_m, train_n] = size(train_data);
%% PCA & SVD
% centralize the data
train_mean = mean(train_data,2); % average by row，1024*1
train_central = train_data - train_mean;
% svd
[U,D,V] = svd(train_central);
% k-nearest neighbor
k = 1;
%% 40
train_40 = U(:,1:40)' * train_data;% 40*2982,每一行是一个样本
test_40 = U(:,1:40)' * test_data;% 40*1278
idx_40 = knnsearch(train_40', test_40');
class_40 = train_label(:,idx_40);
accuracy_pie40 = sum(class_40(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self40 = sum(class_40(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 40 : PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie40*100,accuracy_self40*100);

%% 80
train_80 = U(:,1:80)' * train_data;% 80*2982,每一行是一个样本
test_80 = U(:,1:80)' * test_data;% 80*1278
idx_80 = knnsearch(train_80', test_80');
class_80 = train_label(:,idx_80);
accuracy_pie80 = sum(class_80(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self80 = sum(class_80(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 80 : PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie80*100,accuracy_self80*100);

%% 200
train_200 = U(:,1:200)' * train_data;% 80*2982,每一行是一个样本
test_200 = U(:,1:200)' * test_data;% 80*1278
idx_200 = knnsearch(train_200', test_200');
class_200 = train_label(:,idx_200);
accuracy_pie200 = sum(class_200(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self200 = sum(class_200(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 200: PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie200*100,accuracy_self200*100);
