%% Load MNIST images+labels and display

H = 28; % Image height
W = 28; % Image width
B = 1;  % # of bands (grayscale)
k = 10; % 10 classes (digits 0 to 9)

image_file = 'data/train.images.bin';
label_file = 'data/train.labels.bin';
data_type  = 'uint8';

image_fid = fopen(image_file,'rb');
label_fid = fopen(label_file,'rb');
figure;
for d = 1:25
    I = reshape(fread(image_fid,H*W*B,data_type),W,H,B);
    I = permute(I,[2,1,3]);            % Images are stored row after row
    e = fread(label_fid,k,data_type);  % Label of I in one hot format
    y = find(e)-1;                     % y=0 - first label, y=1 - second label, ...
    subplot(5,5,d);
    imshow(I,[0,255]);
    title(sprintf('label:%d',y));   
end
