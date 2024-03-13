function [ out ] = getLNDPaths( )

    [~, compName] = system('hostname');
    compName = compName(1:(end-1));
    if strcmp(compName, 'Lauras-MacBook-Pro-3.local')
        out.codePath = '/Users/lauradriscoll/Documents/code/yangnet!';
        out.dataPath = '/Users/lauradriscoll/Documents/data/';
    elseif strcmp(compName,'laura-Leopard-WS')
        out.codePath = '/home/laura/code/yangnet';
        out.dataPath = '/home/laura/data';
    end
end

