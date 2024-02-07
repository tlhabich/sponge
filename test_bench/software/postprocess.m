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



% %% Plotte Soll- vs. Istposition
% filename="plots/step_x_d_25.mat";
% load(filename)
% f = figure(1);
% clf(1);
% imesblau   = [0 80 155 ]/255; 
% imesorange = [231 123 41 ]/255; 
% imesgruen  = [200 211 23 ]/255;
% set(f,'DefaultAxesUnit','centimeters')
% set(f,'DefaultAxesFontName','Times')
% set(f,'DefaultAxesFontSize',15)
% set(f,'DefaultAxesTickLabelInterpreter', 'latex')
% set(f,'DefaultLegendInterpreter', 'latex')
% set(f,'defaultTextInterpreter','latex')
% set(f,'DefaultTextFontSize',15)
% 
% f.Units             = 'centimeters';
% f.OuterPosition  	= [30 5 24 16];
% f.Color             = [1 1 1];
% f.PaperSize         = [24 16];
% f.PaperPosition     = [0 0 0 0];
% f.ToolBar           = 'none';
% f.MenuBar           = 'none';
% 
% start=6000;
% n_cut=[start,start+15000];
% PltDat.t=ExpDat.t(n_cut(1):n_cut(2))
% PltDat.x=ExpDat.x(n_cut(1):n_cut(2))
% PltDat.x_d=ExpDat.x_d(n_cut(1):n_cut(2))
% n_plot=300;
% plot_size=size(PltDat.t);
% if plot_size(1)>n_plot
%     idx=round(linspace(1,plot_size(1),n_plot))
%     PltDat.t=PltDat.t(idx)
%     PltDat.x_d=PltDat.x_d(idx)
%     PltDat.x=PltDat.x(idx)
% end
% hold on;
% plot(PltDat.t, PltDat.x,'LineWidth',2,'Color',imesorange);
% plot(PltDat.t, PltDat.x_d, '--','LineWidth',2,'Color',imesblau);
% legend({'$x$', '$x_\mathrm{d}$'});
% xlabel('Zeit in s');
% ylabel('Kolbenposition in mm');
% ylim([22 28])
% grid on
% set(gcf,'PaperPositionMode','auto');
% saveas(gcf,filename+".pdf")