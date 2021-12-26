
path = sprintf('myself');
dir_ls = dir(fullfile(path,'*.jpg'));
num = 10;
idx = 1:num;
shuffle_idx = idx(randperm(num));
img = cell(1, 10);
img_gray = cell(1, 10);
img_resize = cell(1, 10);
figure()
hold on
for j = 1:10
    img{j} = imread(fullfile(path,dir_ls(shuffle_idx(j)).name));
    img_gray{j} = rgb2gray(img{j});
    img_resize{j} = imresize(img_gray{j},[32, 32]);
    % figure()
    subplot(1,10,j);imshow(img{j});
    filename = ['./myself/' num2str(j) '.jpg'];
    imwrite(img_resize{j}, filename);
end  

hold off