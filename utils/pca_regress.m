function [x_reg, y_reg, Beta, coeff] = pca_regress(data,rad_stim)
dim = 3;
%% get ave response
stim_set = unique(rad_stim);

ave_data = [];
for stim = stim_set
    ave_data = cat(2,ave_data,nanmean(data(:,rad_stim==stim),2));
end
%% prep data
data_mean_sub = (ave_data - nanmean(ave_data,2));
coeff = pca(data_mean_sub');

%% Regression
pcs = 1:dim;
X = (coeff(:,pcs)' * data_mean_sub)';
Y = [cos(stim_set)' sin(stim_set)'];
Beta = (X'*X)\(X'*Y);

%% x,y output
x_reg = X*Beta(:,1);
y_reg = X*Beta(:,2);


end

