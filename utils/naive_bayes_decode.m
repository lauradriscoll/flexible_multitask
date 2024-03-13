function frac_corr = naive_bayes_decode(context1,context2,all_data,trial,c)

align = {'Prev End';'Stim';'Delay';'Go';'Move';'End'};
bin_size = 10;
window_size = 80;
ind_set = {[0; trial.end(1:end-1)],trial.stimOn,trial.delayOn,trial.go,trial.moveOn,trial.end};
frac_corr = cell(size(align));

for a = 1:size(align,1)
    
    center_ind = ind_set{a};
    frac_corr{a} = nan(window_size*2,1);
    
    for shift_ind = 1:window_size*2
        g_vec = (trial.task==context1 | trial.task==context2) & ~isnan(center_ind);
        data_vec = nan(size(trial.task,1),size(all_data,2));
        
        for t = find(g_vec)'
            
            ii = ceil(center_ind(t)/20) - window_size + shift_ind;
            
            if a==1
                log = ii>0 & (ii+bin_size) < ind_set{a+1}(t)/20;
            elseif a==size(align,1)
                log = ii > ind_set{a-1}(t)/20;
            else
                log = ii > ind_set{a-1}(t)/20 && (ii+bin_size) < ind_set{a+1}(t)/20;
            end
            
            if log
                inds = ii:(ii+bin_size);
                data_vec(t,:) = nanmean(all_data(inds,:),1);
            else
                g_vec(t) = 0;
            end
            
        end
        
        if sum(g_vec)>80
            allFeatures = data_vec(g_vec,:);
            trlCodes = c(g_vec);
            codeList = unique(trlCodes);
            
            obj = fitcdiscr(allFeatures,trlCodes,'DiscrimType','diaglinear','Prior',ones(size(codeList)));
            cvmodel = crossval(obj);
            predLabels = kfoldPredict(cvmodel);
            
            frac_corr{a}(shift_ind,1) = sum((trlCodes - predLabels)==0)/size(trlCodes,1);
        end
    end
end

end