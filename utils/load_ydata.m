function [R, stream] = load_ydata(YYMMDD,block)

YY = YYMMDD(1:2);
MM = YYMMDD(3:4);
DD = YYMMDD(5:6);


data_dir = fullfile('/Users/lauradriscoll/Documents/data/human/yangnet/',['20' YYMMDD]);

% if strcmp(YYMMDD, '180730')
    ydata = load(fullfile(data_dir,['20' YY '.' MM '.' DD '_block' num2str(block) '.mat']));
    stream = ydata.streams;
    R = ydata.binnedRstream;
% else
%     load(fullfile(data_dir,['formatted' num2str(block) '.mat']));
%     eval(['stream = block' num2str(block) 'stream;'])
%     eval(['R = block' num2str(block) 'binnedR;'])
% end

% spikeRasterCombined = cat(2,stream{1}.spikeRaster,stream{1}.spikeRaster2);
end