function [gbs,fitness_gbs,gbFit_vector]=BpsoFeatSelection(num_ind,max_iter,n,t0,tq,w1,w2,w3,C,gamma,DB,PUWaveDB)

filterName = 'nenhumFilt';
methodName = ['MultFeatures_DS1_' num2str(n) '_' num2str(t0) '_' num2str(tq) '_' filterName]

savefile = strcat('./temp/', methodName,'.mat');
[stat, ~]=fileattrib(savefile);
if stat == 1
    load(savefile);
    display('*** Carrega vetor de VCG em formato MAT');
else
    % percorre a lista de registros para o processo de extra��o de
    % CARACTER�STICAS
    FV=[]; % vetor de features de toda a base
    disp('Extrai caracter�sticas com VCG 3D...');
    tic
    DS1 = [2;5;6;7;9;11;12;13;15;16;18;20;22;24;25;26;27;28;33;35;38;40];   
    parfor i=1:size(DB,2)
        DB(1,i).record
            % extrai apenas para DS1
            if find(DS1==i) > 0
                tic
                FV(i).featureVector = MultipleFeatureExtraction_VCG3D(DB(1,i).ecgDII, DB(1,i).ecgDV1, DB(1,i).anns, DB(1,i).annsPUWave, n, t0, tq, PUWaveDB(i).beat); 
                toc
            else
                FV(i).featureVector = [];
            end
    end
    disp('tempo total pra extracao : ');
    toc
    save(savefile, 'FV');
end

%% PSO
tam_vet=size(FV(2).featureVector,2)-1; %numero de caracteristicas a serem analisadas

%##############------parametros------################
c1=2.05; %peso do pbest
c2=2.05; %peso do gbest
limite_velo=4;
%####################################################

%############    populacao inicial    ###############
particula(1,1:tam_vet) = 1; %Uma particula com todas as features

for ii=2:num_ind
   aux=randi([1,tam_vet]);
    for jj=1:aux
        particula(ii,randi([1,tam_vet]))= 1;
    end    
end

%####################################################



%############----velocidade inicial-----#############
velocidade = (rand(num_ind,tam_vet)*2-1);
%####################################################

plotar=0; %1 para plotar fitness

%##############------parametros------################
wini=0.9;
wend=0.2;
%####################################################


%-----
iter=0;
m=0;
fitness_gbs=0;
fitness_melhor=0;
gbs=zeros(1,tam_vet);
fitness_pbs=zeros(1,num_ind);
pbs=zeros(num_ind,tam_vet);
gbFit_vector=[];


while (iter<max_iter)
    iter=iter+1;
    parfor i=1:(num_ind)
        
        fprintf('\nAvaliando particula %i da iteracao %i ... \n',i,iter)
        tic
        %#############  funcao fitness  ###################
        fitness(i)=classify_ds11_ds12_pesos(FV, methodName, particula(i,:), w1,w2,w3)
        %#############------------------###################
        fprintf('\nTempo para avaliar particula %i: \n',i)
        toc
    end
    for i=1:(num_ind)      
        %seleciona PBest
        if fitness(i)>=fitness_pbs(i)
            fitness_pbs(i)=fitness(i);
            pbs(i,:)=particula(i,:);
        end
        
        %seleciona GBest
        if (fitness_pbs(i)>=fitness_gbs)
            fitness_gbs=fitness_pbs(i);
            gbs=pbs(i,:);
            gbFit_vector=[gbFit_vector fitness_gbs];
        end
         
        
        fprintf('\nAcuracia da particula %i da iteracao %i :  %f \n',i,iter,fitness_pbs(i));
        
        
        
        %###########   calculo da velocidade  ############
        if iter>1
            velocidade(i,:)=velocidade(i,:)+c1*rand()*(pbs(i,:)-particula(i,:))+c2*rand()*(gbs-particula(i,:));
            
        end
        
        
        for j=1:tam_vet
            
            %limite velocidade
            
            if velocidade(i,j)>limite_velo
                velocidade(i,j)=limite_velo;
            end
            if velocidade(i,j)<-limite_velo
                velocidade(i,j)=-limite_velo;
            end
            
            
            %calculo da sigmoide
            s=1/(1+exp(-velocidade(i,j)));
            
            %troca de bit
            if rand()<s
                particula(i,j)=1;
            else
                particula(i,j)=0;
            end
            
        end
        
        %###########   grafico   ##############
        if plotar==1
            plot(iter,fitness_gbs,'y*');
            drawnow;
            hold on;
        end
        %########################################
    end
 save('estadoBPsoFeatSelection', '-regexp', '^(?!(DB|PUWaveDB)$).')    
    
end

end
            
