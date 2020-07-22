% aplica filtro passa-baixa (-3db em 35Hz) 12-taps e passa alta 1Hz
function [filteredDII, filteredDV1] = applyFilters(ecgDII, ecgDV1)
% --------------- Filtra o Sinal de ECG
    % --------------- Filtra o Sinal de ECG
    %aplica filtro passa baixas (-3db em 35Hz) 12-taps
    LPdII = lowPassFIR35;
    ecgDII = LPdII.filter(ecgDII);
    LPdv1 = lowPassFIR35;
    ecgDV1 = LPdv1.filter(ecgDV1);     

    % remove o atraso inserido pelo filtro FIR
    filteredDII=[ecgDII(7:end)];
    filteredDV1=[ecgDV1(7:end)];

    highpass=1;
    if(highpass)
        %aplica filtro passa altas
        HPdII = highPassFIR10;
        filteredDII = HPdII.filter(filteredDII);
        HPdv1 = highPassFIR10;
        filteredDV1 = HPdv1.filter(filteredDV1);

        % remove o atraso inserido pelo filtro FIR e completa com zero
        filteredDII=[filteredDII(41:end)];
        filteredDV1=[filteredDV1(41:end)];
    end
    
end