clear
clc
load('../facedata.mat');
%   train_data  (1024,2982)
%   train_label (1,2982)
%   test_data   (1024,1278)
%   test_label  (1,1278)

%% libSVM
addpath('libsvm-3.25/matlab')

%% PCA & SVD
% centralize the data
train_mean = mean(train_data,2); % average by row，1024*1
train_central = train_data - train_mean;
% svd
[U,D,V] = svd(train_central);

%% original
% model = libsvmtrain(training_label_vector, training_instance_matrix ['libsvm_options']);
model_1 = svmtrain(train_label', train_data', '-t 0 -c 1');
model_01 = svmtrain(train_label', train_data', '-t 0 -c 0.1');
model_001 = svmtrain(train_label', train_data', '-t 0 -c 0.01');

% [predicted_label, accuracy, decision_values/prob_estimates] 
% = libsvmpredict(testing_label_vector, testing_instance_matrix, model ['libsvm_options']);
[~, accuracy_1, ~] = svmpredict(test_label', test_data', model_1); 
[~, accuracy_01, ~] = svmpredict(test_label', test_data', model_01); 
[~, accuracy_001, ~] = svmpredict(test_label', test_data', model_001);

%% dimensionality of 80
train_80 = U(:,1:80)'*train_data; % 80x2982
test_80 = U(:,1:80)'*test_data; % 80x1278
model_80_1 = svmtrain(train_label', train_80', '-t 0 -c 1');
model_80_01 = svmtrain(train_label', train_80', '-t 0 -c 0.1');
model_80_001 = svmtrain(train_label', train_80', '-t 0 -c 0.01');

[~, accuracy_80_1, ~] = svmpredict(test_label', test_80', model_80_1); 
[~, accuracy_80_01, ~] = svmpredict(test_label', test_80', model_80_01); 
[~, accuracy_80_001, ~] = svmpredict(test_label', test_80', model_80_001); 

%% dimensionality of 200
train_200 = U(:,1:200)'*train_data; % 200x2982
test_200 = U(:,1:200)'*test_data; % 200x1278
model_200_1 = svmtrain(train_label', train_200', '-t 0 -c 1');
model_200_01 = svmtrain(train_label', train_200', '-t 0 -c 0.1');
model_200_001 = svmtrain(train_label', train_200', '-t 0 -c 0.01');

[~, accuracy_200_1, ~] = svmpredict(test_label', test_200', model_200_1); 
[~, accuracy_200_01, ~] = svmpredict(test_label', test_200', model_200_01); 
[~, accuracy_200_001, ~] = svmpredict(test_label', test_200', model_200_001); 

%% print accuracy
fprintf('D = 1024:\nc = 1    acc:%.2f%%\nc = 0.1  acc:%.2f%%\nc = 0.01 acc:%.2f%%\n',accuracy_1(1,1),accuracy_01(1,1),accuracy_001(1,1));
fprintf('D = 80:\nc = 1    acc:%.2f%%\nc = 0.1  acc:%.2f%%\nc = 0.01 acc:%.2f%%\n',accuracy_80_1(1,1),accuracy_80_01(1,1),accuracy_80_001(1,1));
fprintf('D = 200:\nc = 1    acc:%.2f%%\nc = 0.1  acc:%.2f%%\nc = 0.01 acc:%.2f%%\n',accuracy_200_1(1,1),accuracy_200_01(1,1),accuracy_200_001(1,1));
