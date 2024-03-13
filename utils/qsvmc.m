function [svm_struct, accuracy_test, accuracy_train] = ...
    qsvmc(data_vec,c1_vec,u_vec,g_vec,C)

%% qsvmc is a function that implements a built-in matlab function, fitcsvm.
% The ouputs are the model struture [svm_struct](see fitcsvm documentation),
% the accuracy in the test group [accuracy_test] and the accuracy of the
% training group [accuracy_train]. The function requires the user to input
% a matrix of data [data_vec](predictor variables by observations, in many
% cases this is cells by trials). A_binary vector of the trial type information
% [c1_vec]. A binary vector of which predictor variables to use, often which 
% cells [u_vec]. A binary vector of which observations to use, often which 
% trials [g_vec]. Lastly, the user must imput the hyperparameter C, which 
% is the only hyperparameter in svmc. The optimal value of C should be
% found in a separate hold out data set using random search or grid search
% for example. I found in most data sets the value of C didn't make a large
% difference and I held it at a constant value for all models.

accuracy_test = nan;
accuracy_train = nan;
data_vec_use = data_vec(u_vec,g_vec);
c1_vec_use = c1_vec(g_vec);

train_inds = randperm(size(data_vec_use,2),round(.75*size(data_vec_use,2)));
test_inds = find(~ismember(1:size(data_vec_use,2),train_inds));

if size(train_inds,2)>10 % require at least 10 observations
    
    % create training and testing subsets
    x_test = data_vec_use(:,test_inds)';
    y_test = c1_vec_use(test_inds);
    
    x_train = data_vec_use(:,train_inds)';
    y_train = c1_vec_use(train_inds);
    
    % make even number of trials from each type 
    [a,b] = hist(y_train,unique(y_train));
    temp1 = find(y_train==b(1));
    temp2 = find(y_train==b(2));
    even_ind = [temp1(randperm(a(1),min(a))) ; temp2(randperm(a(2),min(a)))];
    
    y_train = y_train(even_ind);
    x_train = x_train(even_ind,:);
    
    shuff_ind = randperm(size(y_train,1));
    y_train = y_train(shuff_ind);
    x_train = x_train(shuff_ind,:);
    
    svm_struct = fitcsvm(x_train,y_train,...
        'KernelFunction','linear','BoxConstraint',C);
    train_pred = predict(svm_struct,x_train);
    accuracy_train = nansum(train_pred==y_train)/size(y_train,1);
    test_pred = predict(svm_struct,x_test);
    accuracy_test = nansum(test_pred==y_test)/size(y_test,1);
else
    svm_struct = [];
end
end