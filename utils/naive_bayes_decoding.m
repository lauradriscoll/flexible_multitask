


obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','Prior',ones(length(codeList),1));
cvmodel = crossval(obj);
L = kfoldLoss(cvmodel);
predLabels = kfoldPredict(cvmodel);