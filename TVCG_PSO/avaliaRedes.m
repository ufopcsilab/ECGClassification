%Esta função executa a extração de características de DS1 e classificação com SVM e
%retorna fscore

function fitness=avaliaRedes(individuo,DB,PUWaveDB)

%% Extração de características

n=individuo(1);
t0=individuo(2);
tq=individuo(3);

filterName = 'nenhumFilt'; 
methodName = ['VCG_3D_DS1_' num2str(n) '_' num2str(t0) '_' num2str(tq) '_' filterName]

savefile = strcat('./temp/', methodName,'.mat');
[stat, ~]=fileattrib(savefile);
if stat == 1
    load(savefile);
    display('*** Carrega vetor de VCG em formato MAT');
else
    % percorre a lista de registros para o processo de extraï¿½ï¿½o de
    % CARACTERï¿½STICAS
    FV=[]; % vetor de features de toda a base
    disp('Extrai caracterï¿½sticas com VCG 3D...');
    tic
    DS1 = [2;5;6;7;9;11;12;13;15;16;18;20;22;24;25;26;27;28;33;35;38;40];   
    for i=1:size(DB,2)
        DB(1,i).record
            % extrai apenas para DS1
            if find(DS1==i) > 0
                tic
                FV(i).featureVector = VCG3DFeatureExtraction_PSO(DB(1,i).ecgDII, DB(1,i).ecgDV1, DB(1,i).anns, DB(1,i).annsPUWave, n, t0, tq, PUWaveDB(i).beat);            toc
                toc
            else
                FV(i).featureVector = [];
            end
    end
    disp('tempo total pra extracao : ');
    toc
    save(savefile, 'FV');
end

%% Classificação

disp('Aplica classificador SVM e avalia ...');
tic
%roda o classificador
featSel = []; % nao seleciona caracteristica. Usa todas.
C = 0; % zero para usar valores default
gamma = 0; % zero para usar valores default
fitness=classify_ds11_ds12(FV, methodName, featSel, C, gamma);
toc
end