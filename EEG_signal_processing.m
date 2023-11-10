clc 
clear all
close all

% file path: 
fpath='/Volumes/ELISA/MyelinH/EEG_task/Consegna/';
addpath(genpath(fpath));

data_train=load('B04T.mat');
data_test=load('B04E.mat');

fs=256;

%EEG signals in the training set 
signals_train_1=data_train.data{1,1}.X;
signals_train_2=data_train.data{1,2}.X;
signals_train_3=data_train.data{1,3}.X;

%EEG signals in the test set
signals_test_1=data_test.data{1,1}.X;
signals_test_2=data_test.data{1,2}.X;

%plot the signals (EEG and EOG) in the training set (1st part)
t1=0:1/fs:length(signals_train_1)/fs-1/fs;
figure,
for i=1:3
    subplot(2,3,i), plot(t1,signals_train_1(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EEG signal train_1')
end

for i=4:6
    subplot(2,3,i), plot(t1,signals_train_1(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EOG signal train_1')
end

%% Remove artifacts of EOG signals from the EEG channels

%first part of training set
eeg_data=signals_train_1(:,1:3);
eog_data=signals_train_1(:,4:6);  

% In order to estimate the correction coefficients, I consider the 3rd
% block of the beginning period of each session corresponding to one minute
% with eye artifacts (from minute 3 to minute 4)

t_artifacts=3*60*fs:4*60*fs;

% Create a design matrix with EOG data
X = [ones(size(eog_data(t_artifacts,:),1),1) eog_data(t_artifacts,:)];

% Calculate the regression coefficients for each EEG channel
beta = inv(X'*X)*X'*eeg_data(t_artifacts,:);

% Use the regression model to remove EOG artifacts from the EEG data
eeg_data_clean1 = eeg_data - eog_data*beta(2:end,:);

% Plot the original and cleaned EEG data for comparison
figure,
for i=1:3
    subplot(2,3,i); plot(t1,eeg_data(:,i)); title('Original EEG data'); 
    subplot(2,3,i+3); plot(t1,eeg_data_clean1(:,i)); title('Cleaned EEG data');
end

%% Bandpass filtering between 2-60Hz

order=5;
fband=[2,45]; %Hz
[b,a] = butter(order,fband/(fs/2));

eeg_trainF1=filtfilt(b,a,eeg_data_clean1); %apply the bandpass filter

%plot the PSD of the signals before and after the filtering phase
figure,
%4096*4
[pxx,f0]=pwelch(eeg_data_clean1(:,1),[],[],[],fs);
plot(f0,pxx)
[pxxF,f0]=pwelch(eeg_trainF1(:,1),[],[],[],fs);
hold on, plot(f0,pxxF)
title('Power Spectral Density (PSD) - Welch')
xlabel('frequency (Hz)')
ylabel('power/ frequency (uV^2/Hz)')

figure,
for i=1:3
    subplot(2,3,i); plot(eeg_data_clean1(:,i)); title('Original EEG data');
    subplot(2,3,i+3); plot(eeg_trainF1(:,i)); title('Filtered EEG data');
end

%% Identify and correct the starting points of the trials

start_trial_1=data_train.data{1,1}.trial;
t_trial_1=start_trial_1/fs;

% %plot the signals with the start and ending point of each trial
% figure,
% for i=1:3
%     subplot(2,3,i); plot(t1,eeg_trainF1(:,i)); 
%     hold on, xline(t_trial_1,'r')
%     xlabel('time (s)')
%     ylabel('voltage (uV)')
%     title('EEG signal train_1')
% end

%delete the trials with artifacts
idx1=find(data_train.data{1,1}.artifacts); %index of the trials with artifacts
start_trial_1(idx1)=[]; %the start time of the trials without artifacts

% I decided to select all the 3 s of the imaginary period
im_on=start_trial_1+4*fs; %start of the imaginary period

%% Delete the part of the signals within an imaginary period that are above a given threshold

thr= 60;

% Loop over the channels and time instants (training 3)
for i = 1:size(eeg_trainF1, 2) % Loop over the channels
    for j = 1:length(im_on) % Loop over the time instants
        
        % Find the indices of the time instants within the imaginary period
        indices = im_on(j):(im_on(j)+3*fs); %imaginary period of 3s
        
        % Find the indices of the values above the threshold
        high_indices = find(eeg_trainF1(indices, i) > thr);
        
        % Set the values above the threshold to zero
        eeg_trainF1(indices(high_indices), :) = 0;
        
        % If any values were removed, correct the starting time of the next trials
        if ~isempty(high_indices)
            % Determine the number of time instants to shift the following trials
            shift = length(high_indices);
            
            % Correct the time instants of the following trials
            im_on(j+1:end) = im_on(j+1:end) - shift;
        end
    end
end

im_off=im_on+3*fs-1; %end of the imaginary period (after 3 s)

% Plot the sequences to classify
vuoto=zeros(size(eeg_trainF1));   
figure,
plot(t1,vuoto(:,1))

for m=1:length(im_on)
     hold on
     plot(t1(im_on(m):im_off(m)),eeg_trainF1(im_on(m):im_off(m),1));
end
hold on, yline(50)
xlabel('time (s)')
ylabel('voltage (uV)')
title('Signal ')
% hold on
% xline(t1(im_on),'g')
% hold on 
% xline(t1(im_off),'r')

%% Select the labels and create a struct for the training set 1
label1=data_train.data{1,1}.y;
label1(idx1)=[]; % remove the labels corresponding to the trials with artifacts
label1=label1(:)-1; %final labels (change the labels from 1-2 to 0-1)

training_cell_array1={};
 
for idx = 1:length(im_on)
   training_cell_array1{idx}=eeg_trainF1(im_on(idx):im_off(idx),:);  
end 

% training1=struct('eeg_sequences',training_cell_array1,'label',label1);
% Save the training set1
%save('training1','training1')

%% Repete the same passages for the second and third part of training set 
% and for the test set

%training sets
t2=0:1/fs:length(signals_train_2)/fs-1/fs;
t3=0:1/fs:length(signals_train_3)/fs-1/fs;

%plot signals the 2nd and 3rd part of the trainins set
figure,
for i=1:3
    subplot(4,3,i), plot(t2,signals_train_2(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EEG signal train_2')
    subplot(4,3,i+3), plot(t2,signals_train_2(:,i+3))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EOG signal train_2')
    subplot(4,3,i+6), plot(t3,signals_train_3(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EEG signal train_3')
    subplot(4,3,i+9), plot(t3,signals_train_3(:,i+3))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EOG signal train_3')
end

%test set
t1t=0:1/fs:length(signals_test_1)/fs-1/fs;
t2t=0:1/fs:length(signals_test_2)/fs-1/fs;

%plot signals the 2nd and 3rd part of the test set
figure,
for i=1:3
    subplot(4,3,i), plot(t1t,signals_test_1(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EEG signal test_1')
    subplot(4,3,i+3), plot(t1t,signals_test_1(:,i+3))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EOG signal test_1')
    subplot(4,3,i+6), plot(t2t,signals_test_2(:,i))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EEG signal test_2')
    subplot(4,3,i+9), plot(t2t,signals_test_2(:,i+3))
    xlabel('time (s)')
    ylabel('voltage (uV)')
    title('EOG signal test_2')
end

%% Remove the EOG artifacts using Regression

%training set
eeg_data2=signals_train_2(:,1:3);
eog_data2=signals_train_2(:,4:6);
eeg_data3=signals_train_3(:,1:3);
eog_data3=signals_train_3(:,4:6);
%test set
eeg_data1t=signals_test_1(:,1:3);
eog_data1t=signals_test_1(:,4:6);
eeg_data2t=signals_test_2(:,1:3);
eog_data2t=signals_test_2(:,4:6); 

% In order to estimate the correction coefficients, I consider the 3rd
% block of the beginning period of each session corresponding to one minute
% with eye artifacts (from minute 3 to minute 4)  

% period for estimating the coefficients during eye artifacts
t_artifacts=3*60*fs:4*60*fs;

% Create a design matrix with EOG data
X2 = [ones(size(eog_data2(t_artifacts,:),1),1) eog_data2(t_artifacts,:)];
X3 = [ones(size(eog_data3(t_artifacts,:),1),1) eog_data3(t_artifacts,:)];
X1t = [ones(size(eog_data1t(t_artifacts,:),1),1) eog_data1t(t_artifacts,:)];
X2t = [ones(size(eog_data2t(t_artifacts,:),1),1) eog_data2t(t_artifacts,:)];

% Calculate the regression coefficients for each EEG channel
beta2 = inv(X2'*X2)*X2'*eeg_data2(t_artifacts,:);
beta3 = inv(X3'*X3)*X3'*eeg_data3(t_artifacts,:);
beta1t = inv(X1t'*X1t)*X1t'*eeg_data1t(t_artifacts,:);
beta2t = inv(X2t'*X2t)*X2t'*eeg_data2t(t_artifacts,:);

% Use the regression model to remove EOG artifacts from the EEG data
eeg_data_clean2 = eeg_data2 - eog_data2*beta2(2:end,:);
eeg_data_clean3 = eeg_data3 - eog_data3*beta3(2:end,:);
eeg_data_clean1t = eeg_data1t - eog_data1t*beta1t(2:end,:);
eeg_data_clean2t = eeg_data2t - eog_data2t*beta2t(2:end,:);

%Plot the original and cleaned EEG data for comparison
figure,
for i=1:3
    subplot(2,3,i); plot(eeg_data2t(:,i)); title('Original EEG data');
    subplot(2,3,i+3); plot(eeg_data_clean2t(:,i)); title('Cleaned EEG data');
end

%% Bandpass filtering between 2-60Hz
order=5;
fband=[2,45]; %Hz
[b,a] = butter(order,fband/(fs/2));
%figure,freqz(b,a)
 
% padSize=50;
% engP=[zeros(size(signals_train_1,1),padSize),signals_train_1];
eeg_trainF2=filtfilt(b,a,eeg_data_clean2);
eeg_trainF3=filtfilt(b,a,eeg_data_clean3);
eeg_testF1=filtfilt(b,a,eeg_data_clean1t);
eeg_testF2=filtfilt(b,a,eeg_data_clean2t);

% plot the PSD before an after the filtering phase
figure,
[pxx2,f02]=pwelch(eeg_data_clean2(:,1),[],[],[],fs);
subplot(2,2,1), plot(f02,pxx2), title('Power Spectral Density (PSD) - Welch, signal training_2')
[pxx2,f02]=pwelch(eeg_trainF2(:,1),[],[],[],fs);
hold on, plot(f02,pxx2)

[pxx3,f03]=pwelch(eeg_data_clean3(:,1),[],[],[],fs);
subplot(2,2,2), plot(f03,pxx3), title('Power Spectral Density (PSD) - Welch, signal training_3')
[pxx3,f03]=pwelch(eeg_trainF3(:,1),[],[],[],fs);
hold on, plot(f03,pxx3)

[pxx,f0]=pwelch(eeg_data_clean1t(:,1),[],[],[],fs);
subplot(2,2,3), plot(f0,pxx), title('Power Spectral Density (PSD) - Welch, signal test_1')
[pxx,f0]=pwelch(eeg_testF1(:,1),[],[],[],fs);
hold on, plot(f0,pxx)

[pxx1,f01]=pwelch(eeg_data_clean2t(:,1),[],[],[],fs);
subplot(2,2,4), plot(f01,pxx1), title('Power Spectral Density (PSD) - Welch, signal test_2')
[pxx1,f01]=pwelch(eeg_testF2(:,1),[],[],[],fs);
hold on, plot(f01,pxx1)
xlabel('frequency (Hz)')
ylabel('power/ frequency (uV^2/Hz)')

%% Identify and correct the starting points of the trials

start_trial_2=data_train.data{1,2}.trial;
start_trial_3=data_train.data{1,3}.trial;
start_trial_1t=data_test.data{1,1}.trial;
start_trial_2t=data_test.data{1,2}.trial;

%delete the trials with artifacts
idx2=find(data_train.data{1,2}.artifacts); %index of the trials with artifacts
idx3=find(data_train.data{1,3}.artifacts);
idx1t=find(data_test.data{1,1}.artifacts); 
idx2t=find(data_test.data{1,2}.artifacts);

start_trial_2(idx2)=[]; %the start time of the trials without artifacts
start_trial_3(idx3)=[];
start_trial_1t(idx1t)=[];
start_trial_2t(idx2t)=[];

% I decided to select all the 3s of the imaginary period, which starts
% after 4s of the starting point of each trial
im_on2=start_trial_2+4*fs; %start of the imaginary period
im_on3=start_trial_3+4*fs;
im_on1t=start_trial_1t+4*fs;
im_on2t=start_trial_2t+4*fs;

%% Delete the part of the signals within an imaginary period that are above a given threshold
thr = 60;

% Loop over the channels and time instants (training 2)
for i = 1:size(eeg_trainF2, 2) % Loop over the channels
    for j = 1:length(im_on2) % Loop over the time instants
        
        % Find the indices of the time instants within the imaginary period
        indices = im_on2(j):(im_on2(j)+3*fs); %imaginary period of 3s
        
        % Find the indices of the values above the threshold
        high_indices = find(eeg_trainF2(indices, i) > thr);
        
        % Set the values above the threshold to zero
        eeg_trainF2(indices(high_indices), :) = 0;
        
        % If any values were removed, correct the starting time of the next trials
        if ~isempty(high_indices)
            % Determine the number of time instants to shift the following trials
            shift = length(high_indices);
            
            % Correct the time instants of the following trials
            im_on2(j+1:end) = im_on2(j+1:end) - shift;
        end
    end
end

% Loop over the channels and time instants (training 3)
for i = 1:size(eeg_trainF3, 2) % Loop over the channels
    for j = 1:length(im_on3) % Loop over the time instants
        
        % Find the indices of the time instants within the imaginary period
        indices = im_on3(j):(im_on3(j)+3*fs); %imaginary period of 3s
        
        % Find the indices of the values above the threshold
        high_indices = find(eeg_trainF3(indices, i) > thr);
        
        % Set the values above the threshold to zero
        eeg_trainF3(indices(high_indices), :) = 0;
        
        % If any values were removed, correct the starting time of the next trials
        if ~isempty(high_indices)
            % Determine the number of time instants to shift the following trials
            shift = length(high_indices);
            
            % Correct the time instants of the following trials
            im_on3(j+1:end) = im_on3(j+1:end) - shift;
        end
    end
end

% Loop over the channels and time instants (test 1)
for i = 1:size(eeg_testF1, 2) % Loop over the channels
    for j = 1:length(im_on1t) % Loop over the time instants
        
        % Find the indices of the time instants within the imaginary period
        indices = im_on1t(j):(im_on1t(j)+3*fs); %imaginary period of 3s
        
        % Find the indices of the values above the threshold
        high_indices = find(eeg_testF1(indices, i) > thr);
        
        % Set the values above the threshold to zero
        eeg_testF1(indices(high_indices), :) = 0;
        
        % If any values were removed, correct the starting time of the next trials
        if ~isempty(high_indices)
            % Determine the number of time instants to shift the following trials
            shift = length(high_indices);
            
            % Correct the time instants of the following trials
            im_on1t(j+1:end) = im_on1t(j+1:end) - shift;
        end
    end
end

% Loop over the channels and time instants (test 2)
for i = 1:size(eeg_testF2, 2) % Loop over the channels
    for j = 1:length(im_on2t) % Loop over the time instants
        
        % Find the indices of the time instants within the imaginary period
        indices = im_on2t(j):(im_on2t(j)+3*fs); %imaginary period of 3s
        
        % Find the indices of the values above the threshold
        high_indices = find(eeg_testF2(indices, i) > thr);
        
        % Set the values above the threshold to zero
        eeg_testF2(indices(high_indices), :) = 0;
        
        % If any values were removed, correct the starting time of the next trials
        if ~isempty(high_indices)
            % Determine the number of time instants to shift the following trials
            shift = length(high_indices);
            
            % Correct the time instants of the following trials
            im_on2t(j+1:end) = im_on2t(j+1:end) - shift;
        end
    end
end

% Set the end points of each imaginary period (I decided to keep all the 3s
im_off2=im_on2+3*fs-1; %end of the imaginary period
im_off3=im_on3+3*fs-1;
im_off1t=im_on1t+3*fs-1;
im_off2t=im_on2t+3*fs-1;

% Plot the sequences to classify (only for one subset)
vuoto=zeros(size(eeg_trainF3));   
figure,
plot(t3,vuoto(:,1))

% plot sequences (training set 3)
for m=1:length(im_on3)
     hold on
     plot(t3(im_on3(m):im_off3(m)),eeg_trainF3(im_on3(m):im_off3(m),1));
end
hold on, yline(50)
xlabel('time (s)')
ylabel('voltage (uV)')
title('Signal ')
% hold on
% xline(t1(im_on),'g')
% hold on 
% xline(t1(im_off),'r')

%% Select the labels 

label2=data_train.data{1,2}.y; %select the labels
label3=data_train.data{1,3}.y;
label1t=data_test.data{1,1}.y;
label2t=data_test.data{1,2}.y;

label2(idx2)=[]; %delete the labels corresponding to the trials with artifacts
label3(idx3)=[];
label1t(idx1t)=[];
label2t(idx2t)=[];

label2=label2(:)-1; %final labels (change the labels from 1-2 to 0-1)
label3=label3(:)-1;
label1t=label1t(:)-1;
label2t=label2t(:)-1;

%% Create and save the final struct for the training and test set  

training_cell_array2={};
training_cell_array3={};
test_cell_array1={};
test_cell_array2={};
 
for idx = 1:length(im_on2)
   training_cell_array2{idx}=eeg_trainF2(im_on2(idx):im_off2(idx),:);  
end 

for idx = 1:length(im_on3)
   training_cell_array3{idx}=eeg_trainF3(im_on3(idx):im_off3(idx),:);  
end 

for idx = 1:length(im_on1t)
   test_cell_array1{idx}=eeg_testF1(im_on1t(idx):im_off1t(idx),:);  
end 

for idx = 1:length(im_on2t)
   test_cell_array2{idx}=eeg_testF2(im_on2t(idx):im_off2t(idx),:);  
end 

% concatenate the single subsets 
train_array=cat(2,training_cell_array1, training_cell_array2, training_cell_array3);
test_array=cat(2,test_cell_array1, test_cell_array2);
labels_train=[label1;label2;label3];
labels_test=[label1t;label2t];

training_set=struct('eeg_sequences',train_array,'label',labels_train);
test_set=struct('eeg_sequences',test_array,'label',labels_test);

% Save the datasets for the deep learning model
save('training_set_new','training_set')
save('test_set_new','test_set')

%% Feature extraction

% Define frequency bands for BP calculation
fb_width = [2, 4, 6]; % width of frequency bands in Hz
fb_overlap = 1; % overlap of frequency bands in Hz
freq_bands = [];
f1= 8; % lower limit of frequency in Hz
f2= 30; % upper limit of frequency in Hz
for i = 1:length(fb_width)
    fb_edges = f1:fb_overlap:f2-fb_width(i);
    freq_bands = [freq_bands; fb_edges', fb_edges'+fb_width(i)];
end
n_freq_bands = size(freq_bands, 1);

%% TRAINING SET FEATURE EXTRACTION

% Preallocate variables for BP features
n_samples = size(train_array, 2);
n_channels = 3;
bp_features = zeros(n_samples, n_channels*n_freq_bands);

%initialize a cell array to store the names of the features
col_names = cell(1, n_channels*n_freq_bands);
index = 1;

% Iterate over channels and frequency bands to extract BP features
for eeg =1:n_samples
    for ch = 1:n_channels
        for fb = 1:n_freq_bands % Iterate over frequency bands

            %compute the bandpower 
            eeg_sample=cell2mat(train_array(eeg));
            power_signal= bandpower(eeg_sample(:, ch), fs, freq_bands(fb,:));
            bp_feat = power_signal;

            % Store BP feature in output matrix
            bp_features(eeg, (ch-1)*n_freq_bands+fb) = bp_feat;

            if(eeg==1)
                col_names{index} = sprintf('%d-%d_Hz_ch%d', freq_bands(fb,1), freq_bands(fb,2), ch);
                index = index + 1;
            end
        end
    end
end


Training_features = array2table(bp_features, 'VariableNames', col_names);

train_final = addvars(Training_features, labels_train, 'After', '24-30_Hz_ch3', 'NewVariableName', 'label');


% Save the traing set with features as a file .csv
writetable(train_final, 'trainset_feat_new.csv');

%% TEST SET FEATURE EXTRACTION

% Preallocate variables for BP features
n_samples = size(test_array, 2);
n_channels = 3;
bp_features = zeros(n_samples, n_channels*n_freq_bands);

%initialize a cell array to store the names of the features
col_names = cell(1, n_channels*n_freq_bands);
index = 1;

% Iterate over channels and frequency bands to extract BP features
for eeg =1:n_samples
    for ch = 1:n_channels
        for fb = 1:n_freq_bands % Iterate over frequency bands

            %compute the bandpower 
            eeg_sample=cell2mat(test_array(eeg));
            power_signal= bandpower(eeg_sample(:, ch), fs, freq_bands(fb,:));
            bp_feat = power_signal;

            % Store BP feature in output matrix
            bp_features(eeg, (ch-1)*n_freq_bands+fb) = bp_feat;

            if(eeg==1)
                col_names{index} = sprintf('%d-%d_Hz_ch%d', freq_bands(fb,1), freq_bands(fb,2), ch);
                 index = index + 1;
            end
        end
    end
end


Test_features = array2table(bp_features, 'VariableNames', col_names);

test_final = addvars(Test_features, labels_test, 'After', '24-30_Hz_ch3', 'NewVariableName', 'label');


% Save the traing set with features as a file .csv
writetable(test_final, 'testset_feat_new.csv');

