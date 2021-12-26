clear
clc
% load data
%   train_data  (1024,2982)
%   train_label (1,2982)
%   test_data   (1024,1278)
%   test_label  (1,1278)
load('../facedata.mat');

%% LDA
label_unique = unique(train_label);
avg = mean(train_data,2);
avg_i = cell(1,26);
train = cell(1,26);
s_i = cell(1,26);
num = cell(1,26);
sw = zeros(1024,1024);
sb = zeros(1024,1024);
label_unique = unique(train_label);
for i = 1:26
    ind = find(train_label == label_unique(i));
    train{i} = train_data(:,ind);
    avg_i{i} = mean(train{i},2);
    num{i} = size(train{i}, 2);
    s_i{i} = (train{i}-avg_i{i})*(train{i}-avg_i{i})'/num{i};
    sw = sw + s_i{i}*num{i}/500;
	sb = sb + (avg_i{i}-avg)*(avg_i{i}-avg)'*num{i}/500;
end

[W,Lam] = eig(sb,sw);
all_eigen_values = sum(Lam, 1);
[~, I] = sort(all_eigen_values, 'descend');
W = W(:, I); % 1024*1024

%% 2d
train_2d = W(:,1:2)' * train_data; 
test_2d = W(:,1:2)' * test_data; 
idx_2d = knnsearch(train_2d', test_2d');
class_2d = train_label(:,idx_2d);
accuracy_pie2d = sum(class_2d(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self2d = sum(class_2d(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 2 : PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie2d*100,accuracy_self2d*100);

%% 3d
train_3d = W(:,1:3)' * train_data; 
test_3d = W(:,1:3)' * test_data; 
idx_3d = knnsearch(train_3d', test_3d');
class_3d = train_label(:,idx_3d);
accuracy_pie3d = sum(class_3d(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self3d = sum(class_3d(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 3 : PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie3d*100,accuracy_self3d*100);

%% 9d
train_9d = W(:,1:9)' * train_data; 
test_9d = W(:,1:9)' * test_data; 
idx_9d = knnsearch(train_9d', test_9d');
class_9d = train_label(:,idx_9d);
accuracy_pie9d = sum(class_9d(:,1:1275)==test_label(:,1:1275),'all')/1275;
accuracy_self9d = sum(class_9d(:,1276:1278)==test_label(:,1276:1278),'all')/3;
fprintf('D = 9 : PIE:%.2f%% SELF:%.2f%% \n',accuracy_pie9d*100,accuracy_self9d*100);

