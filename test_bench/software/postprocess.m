% Nachverarbeitung der Modellausgabe aus dem External Mode
% Liest die fortlaufend nummerierten Messdaten ein
% Speichert sie als eine einzige mat-Datei ab
% 
% Abhängigkeit: simulink_signal2struct (imes-matlab-toolbox)

clc
clear
%% Initialisierung
% Verzeichnis, das in Simulink für die Ergebnis-Dateien eingestellt ist
% (Code -> External Mode Control Panel -> Data Archiving)
resdir = fullfile(which(fileparts('postprocess.m')), 'results');
% Ergebnis-Datei
datastructpath = fullfile(resdir, 'measurements_struct.mat');

ExpDat = struct('t', [],"p_bar", [], 'q_deg', [], 'qd_degs', [], 'p_d_bar', []);

%% Daten einlesen
% Gehe alle Dateien durch und fülle die Ergebnis-Struktur
matdatlist = dir(fullfile(resdir, 'measurement_data_*.mat'));

% Suche erste Datei (die Dateien fangen nicht bei 0 an, falls ein
% Ausschnitt der Messdateien irgendwo anders hinkopiert wird)
% Bei Start einer neuen Messung wird auch hochgezählt.
I = -1;
for i = 0:10000
  if exist(fullfile(resdir,sprintf('measurement_data_%d.mat', i)), 'file')
    I = i;
    break;
  end
end
if I == -1
  error('Keine Messwertdateien mit Namensschema gefunden');
end
for i = 1:length(matdatlist)
  % neuen Dateinamen (die Liste ist nicht unbedingt alphabetisch)
  dateiname_neu = sprintf('measurement_data_%d.mat', I+i-1);
  fprintf('Lese Datei %d/%d: %s.\n', i,length(matdatlist), dateiname_neu);
    
  % Datei öffnen
  matdatpath = fullfile(resdir, dateiname_neu);
  tmp = load(matdatpath);
  
  % Daten konvertieren in Datenstruktur mit nur einem einzigen Zeitfeld
  % (ist wesentlich übersichtlicher als Simulink-Struktur DataWithTime
  sl_signal = simulink_signal2struct(tmp.ScopeData1);

  % An Struktur anhängen
  ExpDat = timestruct_append(ExpDat, sl_signal);
end
% Speichere die fertig formatierten Daten
% Nutze Format 7.3, damit große Datenreihen auch gespeichert werden können
save(datastructpath, 'ExpDat', '-v7.3');