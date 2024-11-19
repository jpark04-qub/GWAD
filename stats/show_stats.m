clear all
close all


%%
dir = '';

[FileName, PathName, FilterIndex]= uigetfile([dir '*.txt'],'MultiSelect','on');

number = 1;
if iscell(FileName)
    number = length(FileName);
end
for i = 1:number
    if iscell(FileName)
        feature = load([PathName, FileName{i}]);    
        FeatureName = FileName{i}(1:end-4);
    else
        feature = load([PathName, FileName]);
        FeatureName = FileName(1:end-4);
    end

    figure(i);
    if contains(PathName, 'hist')
        x = 1:length(feature);
        bar(x, feature);
    else
        if size(feature, 2) == 1
            x = 1:length(feature);
            plot(x, feature, '.');
            ylim([-1.1, 1.1]);        
        else
            clim([-1.1, 1.1]);
            imagesc(feature, clim);
        end
    end
    title(FeatureName);
    set(gcf,'color','w');
    set(gca,'XTick',[], 'YTick', []);

    set(gcf, 'Units', 'inches');
    scrpos = get(gcf, 'Position');
    set(gcf, 'PaperPosition',[0 0 scrpos(3:4)], 'PaperSize', [scrpos(3:4)]);
end


