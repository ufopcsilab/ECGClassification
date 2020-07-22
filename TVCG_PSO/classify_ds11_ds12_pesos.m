function fitness=classify_ds11_ds12_pesos(featureSet, featureString, featSel, w1,w2,w3)
% Classificação de Arritmia . Treina com DS1 e testa com DS2.
%
% >> featureSet: conjunto de características

% >> featSel: vetor para seleção de características
% Deve ser no formato: fs = [1 1 0 0 0 1 0 0 1], do dimensão das
% características onde:
% 1 - usar a característica
% 0 - ignorar a característica
% passe um vetor vazio para utilizar todas as características
%
% >> classifier: classificador a ser utilizado
% Pode assumir os valores:
%
% 'SVM', 'LD', 'MLP'
%
% C: parâmetro para SVM
% 0: escolhe default (0.05)
% -1: utiliza script para grid selection

% gamma: parâmetro para SVM
% 0: escolhe default (1/8*num features)
%
% Autor: Eduardo Luz
%
%

if nargin < 4
    C = 0;
    gamma = 0; % default
elseif nargin < 5
    gamma = 0;
end

s = char(featureString);
numfs = size(featSel,2);
fileNamed = ['./resultados/results_',s,'_SVM_','_DS1_DS2_FS_' num2str(numfs) 'results.tex'];
arq = fopen(fileNamed,'w');

% Tabela latex dos resultados
fprintf(arq,'\\documentclass{article}\n');
fprintf(arq,'\\usepackage{graphicx}\n');
fprintf(arq,'\\usepackage[latin1]{inputenc}\n');
fprintf(arq,'\\usepackage{tabularx}\n');
fprintf(arq,'\\usepackage{multirow}\n');
fprintf(arq,'\\newcommand{\\citep}{\\cite}\n');
fprintf(arq,'\\newcommand{\\citet}{\\cite}\n');
fprintf(arq,'\\newcommand{\\TFigure}{Fig.}\n');
fprintf(arq,'\\begin{document}\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,'\\footnotesize \n');
s2 = char('\\caption{Tabela de resultados por paciente ');
s2 = [s2 s(10:end-1) '} \n'];
fprintf(arq,s2);
%fprintf(arq,' \\caption{Tebela dos registros do método} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|c|c|c|c|c|c|c|} \n');
fprintf(arq,'   \\hline \n');
fprintf(arq,'    Registro & Acc & N Se/+P/FPR & SVEB Se/+P/FPR & VEB Se/+P/FPR & F Se/+P/FPR & Q Se/+P/FPR  \\\\ \n');
fprintf(arq,'   \\hline \n');

% iniciliza variaveis
finalCM = zeros(5,5);
fprintf('\n DS1xDS2 | Acc | N_Se SVEB_Se VEB_Se N+P SVEB+P VEB+P \n')


% Carrega os dados de treino e teste
[p1d, p1t, p2d, p2t] = loadDataAAMI_DS1(featureSet, 0);

% seleciona caracteristicas, se for o caso
if ~isempty(featSel)
    fsel = find(featSel==1);
    p1d = p1d(:,fsel);
    p2d = p2d(:,fsel);
end

%primeira etapa com DS1 treino DS2 teste
fs1.train = double(p1d); clear p1d;
fs1.test = double(p2d); clear p2d;
target.train = double(p1t); clear p1t;
target.test = double(p2t); clear p2t;

% normaliza
[fs1.train, scale_factor] = mapminmax(fs1.train');
fs1.test = mapminmax('apply',fs1.test',scale_factor);

fs1.train = fs1.train';
fs1.test = fs1.test';

%% aplica o classificador

cd('svm_linux')
%cd('svm_win')
%cd('svm_mac')

modsel_weight = ['-w1 ' num2str(w1) ' -w2 ' num2str(w2) ' -w3 ' num2str(w3)];

tic
best_c=0;best_g=0; %nao estao sendo usados
clear cm1;
[cm1] = svm_Classifier(modsel_weight,fs1.train,target.train,fs1.test,target.test,best_c,best_g);

toc

cd ..


%% Calcula estatíSticas
cm1

finalCM = finalCM + cm1;

acc_num=0;
acc_den=0;
den1=0;
den2=0;
num=0;
t=0;

if(size(cm1,1)>=1)
    t = 1;
    
    num = cm1(t,t);
    den1 = sum(cm1(t,:));
    den2 = sum(cm1(:,t));
    
    TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
    FP = den2 - cm1(t,t);
    
    if(den1~=0)
        sensitivityN = (num/den1) * 100;
    else
        sensitivityN = -1;
    end
    
    if(den2~=0)
        specificityN = (num/den2) * 100;
    else
        specificityN = -1;
    end
    
    if(TN + FP > 0)
        FPR_N = 100*FP/(TN+FP);
    else
        FPR_N=-1;
    end
    
    acc_num = acc_num + num;
    acc_den = acc_den + den1;
    
else
    sensitivityN = -1;
    specificityN = -1;
    FPR_N=-1;
end


if(size(cm1,1)>=2)
    t = 2;
    
    num = cm1(t,t);
    den1 = sum(cm1(t,:));
    den2 = sum(cm1(:,t));
    
    TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
    FP = den2 - cm1(t,t);
    
    if(den1~=0)
        sensitivitySVEB = (num/den1) * 100;
    else
        sensitivitySVEB = -1;
    end
    
    if(den2~=0)
        specificitySVEB = (num/den2) * 100;
    else
        specificitySVEB = -1;
    end
    
    if(TN + FP > 0)
        FPR_SVEB = 100*FP/(TN+FP);
    else
        FPR_SVEB=-1;
    end
    
    acc_num = acc_num + num;
    acc_den = acc_den + den1;
    
else
    sensitivitySVEB = -1;
    specificitySVEB = -1;
    FPR_SVEB=-1;
end

if(size(cm1,1)>=3)
    t = 3;
    num = cm1(t,t);
    den1 = sum(cm1(t,:));
    den2 = sum(cm1(:,t));
    
    TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
    FP = den2 - cm1(t,t);
    
    if(den1~=0)
        sensitivityVEB = (num/den1) * 100;
    else
        sensitivityVEB = -1;
    end
    
    if(den2~=0)
        specificityVEB = (num/den2) * 100;
    else
        specificityVEB = -1;
    end
    
    if(TN + FP > 0)
        FPR_VEB = 100*FP/(TN+FP);
    else
        FPR_VEB=-1;
    end
    
    acc_num = acc_num + num;
    acc_den = acc_den + den1;
    
else
    sensitivityVEB = -1;
    specificityVEB = -1;
    FPR_VEB=-1;
end

if(size(cm1,1)>=4)
    t = 4;
    
    num = cm1(t,t);
    den1 = sum(cm1(t,:));
    den2 = sum(cm1(:,t));
    
    TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
    FP = den2 - cm1(t,t);
    
    if(den1~=0)
        sensitivityF = (num/den1) * 100;
    else
        sensitivityF = -1;
    end
    if(den2~=0)
        specificityF = (num/den2) * 100;
    else
        specificityF = -1;
    end
    
    if(TN + FP > 0)
        FPR_F = 100*FP/(TN+FP);
    else
        FPR_F=-1;
    end
    
    acc_num = acc_num + num;
    acc_den = acc_den + den1;
    
else
    sensitivityF = -1;
    specificityF = -1;
    FPR_F=-1;
end

if(size(cm1,1)>=5)
    t = 5;
    
    num = cm1(t,t);
    den1 = sum(cm1(t,:));
    den2 = sum(cm1(:,t));
    
    TN = sum(sum(cm1(:,:))) - den1 - den2 + cm1(t,t);
    FP = den2 - cm1(t,t);
    
    if(den1~=0)
        sensitivityQ = (num/den1) * 100;
    else
        sensitivityQ = -1;
    end
    
    if(den2~=0)
        specificityQ = (num/den2) * 100;
    else
        specificityQ = -1;
    end
    
    if(TN + FP > 0)
        FPR_Q = 100*FP/(TN+FP);
    else
        FPR_Q=-1;
    end
    
    acc_num = acc_num + num;
    acc_den = acc_den + den1;
    
else
    sensitivityQ = -1;
    specificityQ = -1;
    FPR_Q=-1;
end

fprintf(arq,'- & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f \\\\ \n',...
    100*acc_num/acc_den, sensitivityN,specificityN,FPR_N,sensitivitySVEB,specificitySVEB,FPR_SVEB,sensitivityVEB,specificityVEB,FPR_VEB,...
    sensitivityF,specificityF,FPR_F, sensitivityQ,specificityQ,FPR_Q);
fprintf(arq,'\n');

fprintf('\n');

fprintf('Registro= -  | Acc=%6.1f |', 100*acc_num/acc_den);

fprintf(' N_Se=%6.1f SVEB_Se=%6.1f VEB_Se=%6.1f N+P=%6.1f SVEB+P=%6.1f VEB+P=%6.1f \n\n',...
    sensitivityN, sensitivitySVEB,sensitivityVEB,  specificityN, specificitySVEB,specificityVEB)

%% computa estatisticas

fprintf(arq,'    \\hline \n');

TN = sum(sum(finalCM(:,:))) - sum(finalCM(1,:)) - sum(finalCM(:,1)) + finalCM(1,1);
FP = sum(finalCM(:,1)) - finalCM(1,1);
FPR_N = FP/(TN+FP);

TN = sum(sum(finalCM(:,:))) - sum(finalCM(2,:)) - sum(finalCM(:,2)) + finalCM(2,2);
FP = sum(finalCM(:,2)) - finalCM(2,2);
FPR_SVEB = FP/(TN+FP);

TN = sum(sum(finalCM(:,:))) - sum(finalCM(3,:)) - sum(finalCM(:,3)) + finalCM(3,3);
FP = sum(finalCM(:,3)) - finalCM(3,3);
FPR_VEB = FP/(TN+FP);

if size(finalCM,1) > 3
    
    TN = sum(sum(finalCM(:,:))) - sum(finalCM(4,:)) - sum(finalCM(:,4)) + finalCM(4,4);
    FP = sum(finalCM(:,4)) - finalCM(4,4);
    FPR_F = FP/(TN+FP);
    
    TN = sum(sum(finalCM(:,:))) - sum(finalCM(5,:)) - sum(finalCM(:,5)) + finalCM(5,5);
    FP = sum(finalCM(:,5)) - finalCM(5,5);
    FPR_Q = FP/(TN+FP);
    
    fprintf(arq, ' Gross & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f \\\\ \n',...
        100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
        100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),100*FPR_N,...
        100*finalCM(2,2)/sum(finalCM(2,:)),100*finalCM(2,2)/sum(finalCM(1:end-1,2)),100*FPR_SVEB,...
        100*finalCM(3,3)/sum(finalCM(3,:)),100*finalCM(3,3)/sum(finalCM(1:end-2,3)),100*FPR_VEB,...
        100*finalCM(4,4)/sum(finalCM(4,:)),100*finalCM(4,4)/sum(finalCM(:,4)),100*FPR_F,...
        100*finalCM(4,4)/sum(finalCM(5,:)),100*finalCM(5,5)/sum(finalCM(:,5)),100*FPR_Q);
    
else
    fprintf(arq, ' Gross & %6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & %6.1f/%6.1f/%6.1f & - & - \\\\ \n',...
        100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))) ,...
        100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(1,1)/sum(finalCM(:,1)),100*FPR_N,...
        100*finalCM(2,2)/sum(finalCM(2,:)),100*finalCM(2,2)/sum(finalCM(:,2)),100*FPR_SVEB,...
        100*finalCM(3,3)/sum(finalCM(3,:)),100*finalCM(3,3)/sum(finalCM(:,3)),100*FPR_VEB);
end

fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'\n');

fprintf(arq,'\n');
fprintf(arq,'\\begin{table*} \n');
fprintf(arq,' \\caption{Matriz e confusão} \n');
fprintf(arq,' \\label{tab:regtable} \n');
fprintf(arq,' \\begin{center} \n');
fprintf(arq,'  \\begin{tabular}{|');
for tt=1:size(finalCM,1)
    fprintf(arq,'c|');
end
fprintf(arq,'} \n');
fprintf(arq,'\n');
fprintf(arq,'   \\hline \n');
for tt=1:size(finalCM,1)
    for uu=1:size(finalCM,2)
        if uu==5
            fprintf(arq,'%6.0f',finalCM(tt,uu));
        else
            fprintf(arq,'%6.0f & ',finalCM(tt,uu));
        end
    end
    fprintf(arq,'    \\\\ \n');
    fprintf(arq,'   \\hline \n');
end
fprintf(arq,'   \\hline \n');
fprintf(arq,'  \\end{tabular} \n');
fprintf(arq,' \\end{center} \n');
fprintf(arq,'\\end{table*} \n');
fprintf(arq,'\n');
fprintf(arq,'SVM parameters - C = %6.6f, gamma= %6.6f',best_c, best_g);
fprintf(arq,'   \\hline \n');
fprintf(arq,'SVM pesos por classe %s',modsel_weight);
fprintf(arq,'\n');
fprintf(arq,'\\end{document}\n');
fclose(arq);

fprintf('\n Gross Statistics:\n');
fprintf(' Gross & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f & %6.1f \\\\ \n',...
    100*(finalCM(1,1)+finalCM(2,2)+finalCM(3,3)+finalCM(4,4)+finalCM(5,5))/(sum(finalCM(1,:))+sum(finalCM(2,:))+sum(finalCM(3,:))+sum(finalCM(4,:))+sum(finalCM(5,:))) ,...
    100*finalCM(1,1)/sum(finalCM(1,:)), 100*finalCM(2,2)/sum(finalCM(2,:)), 100*finalCM(3,3)/sum(finalCM(3,:)), 100*finalCM(4,4)/sum(finalCM(4,:)), 100*finalCM(5,5)/sum(finalCM(5,:)),...
    100*finalCM(1,1)/sum(finalCM(:,1)), 100*finalCM(2,2)/sum(finalCM(:,2)), 100*finalCM(3,3)/sum(finalCM(:,3)), 100*finalCM(4,4)/sum(finalCM(:,4)), 100*finalCM(5,5)/sum(finalCM(:,5)));

NS=100*finalCM(1,1)/sum(finalCM(1,:));%N SENSITIVITY
NP=100*finalCM(1,1)/sum(finalCM(:,1));%N POSITIVE PREDICTIVE VALUE

SVS=100*finalCM(2,2)/sum(finalCM(2,:));%S SENSITIVITY
SVP=100*finalCM(2,2)/sum(finalCM(1:end-1,2));%S POSITIVE PREDICTIVE VALUE

VES=100*finalCM(3,3)/sum(finalCM(3,:));%V SENSITIVITY
VEP=100*finalCM(3,3)/sum(finalCM(1:end-2,3));%V POSITIVE PREDICTIVE VALUE

%Calcula F-score médio
fitness=fscore(fun_fscore(NS,NP),fun_fscore(SVS,SVP),fun_fscore(VES,VEP));
end

function fs=fun_fscore(S,P)
if S==0 | P==0
    fs=0;
else
    fs=2*((S*P)/(S+P));
end
end