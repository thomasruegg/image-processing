% Task: Bild laden und darstellen.

clear all; close all; clc;
%% bild laden
im = imread('../../images/lions.jpg','jpg');

%% bild anzeigen (verschiedene Varianten)

% Variante 1: braucht uint8 Werte (schwarz=0, ..., weiss=255), 
% jedes pixel wird mit einem Bildschirmpixel dargestellt 
% -> grosse Auflösung gibt grosse Anzeige auf Bildschirm.
figure(1)
imshow(im);

% Variante 2: Beliebiger Wertebereich, schwart = kleinster Wert
%                                      weiss   = grösster Wert
% geht auch mit double-Werten
% sonst wie Variante 1.
figure(2)
imshow(im, []);

% Variante 2: Braucht Werte 0..255, stellt graustuffenbilder in
% Falschfarben dar (falls nicht colormap(gray) gesetzt).
figure(3)
image(im);

% Variante 4: Beliebiger Wertebereich, sonst wie Variante 3.
figure(4)
imagesc(im);
