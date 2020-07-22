function fitness=avaliaFinal_chazal(gbVar_featSel,n,t0,tq,w1,w2,w3,gamma,C,DB,PUWaveDB)

%% ** PR�-PROCESSAMENTO **
filterName = 'FiltChazal';
for i=1:size(DB,2)
        disp('filtra sinal com filtros do de Chazal');
        [filteredDII,filteredDV1] = applyChazalFilters(DB(1,i).ecgDII, DB(1,i).ecgDV1);
        DB(1,i).ecgDII = filteredDII; clear filteredDII;
        DB(1,i).ecgDV1 = filteredDV1; clear filteredDV1;
end
methodName = ['MultFeatures_' num2str(n) '_' num2str(t0) '_' num2str(tq) '_' filterName]
savefile = strcat('./temp/', methodName,'.mat');
[stat, ~]=fileattrib(savefile);
if stat == 1
    load(savefile);
    display('*** Carrega vetor de VCG em formato MAT');
else
    % percorre a lista de registros para o processo de extra��o de
    % CARACTER�STICAS
    FV=[]; % vetor de features de toda a base
    disp('Extrai caracter�sticas com multiplas features...');
    tic
    parfor i=1:size(DB,2)
        DB(1,i).record      
        FV(i).featureVector = MultipleFeatureExtraction_VCG3D(DB(1,i).ecgDII, DB(1,i).ecgDV1, DB(1,i).anns, DB(1,i).annsPUWave, n, t0, tq, PUWaveDB(i).beat);
    end
    disp('tempo total pra extracao : ');
    toc
    save(savefile, 'FV');
end
fitness=classify_ds1_ds2(FV, methodName, gbVar_featSel, C, gamma,w1,w2,w3);
end


