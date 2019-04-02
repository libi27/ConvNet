runs = 1:5;
batch_sizes = [1, 2, 4, 8, 20];
learning_rates = [0.5, 0.1, 0.05];
results = zeros(numel(runs), numel(batch_sizes));
accuracies = zeros(1,numel(learning_rates));


%% Template script for constructing, training and evaluating Convolutional Neural Network

%% Prepare
GPU = false; % Set to true for GPU mode (requires parallel computing toolbox)
% Format
H = 28;      % Image height
W = 28;      % Image width
B = 1;       % # of bands (grayscale)
k = 10;      % 10 classes (digits 0 to 9)
% Train and test files
train_images_file = 'data/train.images.bin';
train_labels_file = 'data/train.labels.bin';

%% Construct: conv-->relu-->max-pool-->affine-->loss
% For technical reasons, data batch size and weight initialization are part of network definition.
for r = runs 
    i = 1;
    for batch_size = batch_sizes
        j = 1;
        for learning_rate = learning_rates
            conv_kernel   = 5;
            conv_stride   = 1;
            conv_channels = 5;
            pool_kernel   = 2;
            pool_stride   = 2;
            net_arch = {...
                struct('type','input','inInd',0,'outInd',1,'blobSize',[H,W,B,batch_size],'fName',train_images_file,'scale',1/256,'dataType','uint8'), ...
                struct('type','input','inInd',0,'outInd',2,'blobSize',[k,batch_size],'fName',train_labels_file,'scale',1,'dataType','uint8'), ...
                struct('type','conv','inInd',1,'outInd',3,'kernelsize',conv_kernel,'stride',conv_stride,'nOutChannels',conv_channels,'bias_filler',0),...
                struct('type','relu','inInd',3,'outInd',3),  ...
                struct('type','maxpool','inInd',3,'outInd',4,'kernelsize',pool_kernel,'stride',pool_stride), ...
                struct('type','flatten','inInd',4,'outInd',4), ...
                struct('type','affine','inInd',4,'outInd',5,'nOutChannels',k,'bias_filler',0), ...
                struct('type','loss','inInd',[5 2],'outInd',6,'lossType','MCLogLoss')};
            net = ConvNet(net_arch,GPU,'Xavier');  % Xavier initialization of conv and affine layers

            %% Train
            % Optimization parameters
            T      = 1e4 / batch_size; % No. of iterations for 4 epochs (10K image passes)
            mu     = single(0);   % Momentum variable
            lambda = single(0);   % Regularization constant
            eta    = @(t)(learning_rate); % Learning rate - eta(t) returns the value for iteration t
            % Display and snapshot parameters
            stat_param.printIter    = 1e2;
            stat_param.printDecay   = 0.9;  % Decay factor - rolling average displayed and saved
            stat_param.snapshotFile = 'snapshots/snapshot';
            % Run SGD with Nesterov momentum

            net.Nesterov(T,eta,mu,lambda,stat_param);

            %% Evaluate
            % Train accuracy
            bad_pred  = 0;
            good_pred = 0;
            for b = 1:net.net{1}.data.m  % Scan through data batches
                net.forward(b);
                net_outputs = net.O{net.net{end}.inInd(1)};
                labels_1hot = net.O{net.net{end}.inInd(2)};
                [~,b_pred ] = max(net_outputs);
                [~,b_label] = max(labels_1hot);
                bad_pred    = bad_pred +sum(b_pred~=b_label);
                good_pred   = good_pred+sum(b_pred==b_label);
            end
            train_acc = good_pred/(good_pred+bad_pred);
            accuracies(j) = train_acc;
            j = j+1;
        end 
        results(r,i) = 1 - max(accuracies);
        i=i+1;
    end 
end 
% results
avgerr = sum(results)./length(runs)

figure;
plot(batch_sizes, avgerr);
xlabel('Batch Size') 
ylabel('Error Measure')

