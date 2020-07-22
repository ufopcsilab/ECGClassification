% aplica filtro proposto por de Chazal et al. (2004)
function [filteredDII, filteredDV1] = applyChazalFilters(ecgDII, ecgDV1)

% primeiro filtro com uma janela de 200ms (aprox. 71 amostras)
ecg200MeanFiltDII=medfilt1(ecgDII,71);
% segundo filtro com uma janela de 600ms (aprox. 214 amostras)
ecg600MeanFiltDII=medfilt1(ecg200MeanFiltDII,214);
% primeiro filtro com uma janela de 200ms (aprox. 71 amostras)
ecg200MeanFiltDV1=medfilt1(ecgDV1,71);
% segundo filtro com uma janela de 600ms (aprox. 214 amostras)
ecg600MeanFiltDV1=medfilt1(ecg200MeanFiltDV1,214);

ecgDII=ecgDII-ecg600MeanFiltDII;    % remove baseline wander
ecgDV1=ecgDV1-ecg600MeanFiltDV1;


filter25=0;
filter35=1;
if(filter25==1)
      % DII
      ECG500dii=interp1(1:size(ecgDII,1),ecgDII,1:360/500:size(ecgDII,1))';

      %Filter then downsample
      lp_500 = lowPass500FIR25;
      ECG500dii = lp_500.filter(ECG500dii);
      ecgDII=interp1(1:size(ECG500dii,1),ECG500dii,1:500/360:size(ECG500dii,1))';      

      %LEAD DV
      ECG500dv1=interp1(1:size(ecgDV1,1),ecgDV1,1:360/500:size(ecgDV1,1))';

      %Filter then downsample
      lp_500dv1 = lowPass500FIR25;
      ECG500dv1 = lp_500dv1.filter(ECG500dv1);
      ecgDV1=interp1(1:size(ECG500dv1,1),ECG500dv1,1:500/360:size(ECG500dv1,1))';

      % remove o atraso inserido pelo filtro FIR e completa com zero
      filteredDII=[zeros(19,1); ecgDII(20:end)];
      filteredDV1=[zeros(19,1); ecgDV1(20:end)];
    end

    if(filter35==1)
        % DII
      ECG500dii=interp1(1:size(ecgDII,1),ecgDII,1:360/500:size(ecgDII,1))';

      %Filter then downsample
      lp_500 = lowpass500FIR35;
      ECG500dii = lp_500.filter(ECG500dii);
      ecgDII=interp1(1:size(ECG500dii,1),ECG500dii,1:500/360:size(ECG500dii,1))';      

      %LEAD DV
      ECG500dv1=interp1(1:size(ecgDV1,1),ecgDV1,1:360/500:size(ecgDV1,1))';

      %Filter then downsample
      lp_500dv1 = lowpass500FIR35;
      ECG500dv1 = lp_500dv1.filter(ECG500dv1);
      ecgDV1=interp1(1:size(ECG500dv1,1),ECG500dv1,1:500/360:size(ECG500dv1,1))';

      % remove o atraso inserido pelo filtro FIR
      filteredDII=[zeros(7,1); ecgDII(8:end)];
      filteredDV1=[zeros(7,1); ecgDV1(8:end)];
    end
    
end