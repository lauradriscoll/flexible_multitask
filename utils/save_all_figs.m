function save_all_figs(savedir)

d = fullfile(savedir);
if ~exist(d,'dir')
    mkdir(d)
end

FolderName = d;   % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
    FigHandle = FigList(iFig);
    FigName   = [string(iFig) '.fig'];
    savefig(fullfile(FolderName, FigName));
    FigName   = [string(iFig) '.pdf'];
    savefig(fullfile(FolderName, FigName));
end
end