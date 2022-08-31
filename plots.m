dirs = ["lc_glove_results",  "lc_m2v_results",  "lc_hybrid_results",...
        "lcc_glove_results", "lcc_m2v_results", "lcc_hybrid_results",...
        "llc_glove_results", "llc_m2v_results", "llc_hybrid_results"
       ];
folds = ["k_fold5", "k_fold"];

for k = 1:length(folds)
    f1 = zeros(9, k*5);
    for d = 1:length(dirs)
        fstruct = dir(strcat("src/", dirs(d), "/", folds(k), "/*.test.metrics*.txt"));
        for i = 1:length(fstruct)
            fid = fopen(strcat(fstruct(i).folder, "/", fstruct(i).name));
            result = fscanf(fid, '%c', 500);
            words = split(result);
            f1_ = words{find(strcmp(words, 'FB1:'))+1};
            f1(d,i) = str2double(f1_);
        end
    end
    f1_mean = mean(f1, 2);
    f1_std = std(f1, 0, 2);
    % 1 - the median
    % 2 - the lower and upper quartiles
    % 3 - any outliers 
    % 4 - minimum and maximum values that are not outliers
    figure;
    boxchart(f1');axis padded; grid on;
    tt = sprintf("F1 metric distribution of %d-fold cross validation", k*5);
    title(tt)
    xticklabels({'lc\_glove','lc\_m2v','lc\_hybrid',...
             'lcc\_glove','lcc\_m2v','lcc\_hybrid',...
             'llc\_glove','llc\_m2v','llc\_hybrid'});
end
    

