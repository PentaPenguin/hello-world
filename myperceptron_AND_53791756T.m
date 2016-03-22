function myperceptron_AND_53791756T
% function myperceptron_AND_53791756T
% Función que crea, entrena y comprueba que un perceptrón se adapta
% a una función lógica AND
% Código completado por David Millán Navarro y Guillermo Moreno Hernández
% Sistemas complejos bioinspirados - ETSIT - UPV - Curso 2015-2016
% Basado en el código de Andreu M. Climent

% Funcion principal que realiza las funciones de
%1) Creación de las variables para el banco de entrenamiento y banco de
 %validación
%2) Creación de la red neuronal (perceptron)
%3) Entrenamiento de la red neuronal con el set de valores del banco de
 %entrenamiento
%4) Validación de la red con el banco de validación.
%5) Calculo y representación del error cometido.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Creamos entradas y salidas de entrenamiento para una funcion AND
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inicializamos las variables E1, E2 y SE ideales para entrenamiento
E1=round(rand(5000,1));
E2=round(rand(5000,1));
SE=E1.*E2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Creamos entradas y salidas de validacion E1V, E2V, SEV para test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
E1V=round(rand(20,1));
E2V=round(rand(20,1));
SEV=E1V.*E2V;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inicializamos un perceptron para 2 entradas %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myperceptron=initialize_perceptron(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Entrenamos el perceptron para un LR, por defecto 0.7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LR=0.7;
myperceptronT=train_perceptron(myperceptron,LR,[E1 E2],SE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluamos el perceptron
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S_est=useperceptron(myperceptronT,[E1V E2V]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculamos y representamos el error cometido
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error=mean(abs(SEV-S_est));
close all
figure,
plot(SEV,'ok','LineWidth',2),hold on
plot(S_est,'xr','LineWidth',2)
%set(gcf,'FontWeight','bold')
set(gca,'FontSize',12) %# Fix font size of the text in the current axes
set(gca,'FontWeight','bold') %# Fix Bold text in the current axes 
xlabel('Number of test','FontWeight','bold')
ylabel('Output values','FontWeight','bold')
axis([-1 length(SEV)+1 -0.1 1.3])
legend('Correct Values','Perceptron Output')
title('Evaluation of Perceptron Ouput for an AND','FontWeight','bold')

end

function [myperceptron]=initialize_perceptron(n_inputs)
    
%function [myperceptron]=initialize_perceptron(n_inputs)
%funcion que inicializa un perceptron y le asigna pesos aleatorios
% Esta funcion crea e inicializa la esctructura myperceptron.weights donde se
% guardan los pesos del perceptron. Los valores de los pesos iniciales
% deben ser numeros aleatorios entre -1 y 1.
% INPUTS:
 % n_inputs: numero de entradas al perceptron
% OUTPUT:
 % myperceptron: estructura con el perceptron ya entrenado
 % myperceptron.bias: %Bias del perceptron (e.g. 1)
 % myperceptron.weights: %Pesos del perceptron
 
 myperceptron.bias=1;
 myperceptron.weights=2*rand(n_inputs+1,1)-1; % Genereamos valores de bias entre -1 y 1
 
end

function myperceptron=train_perceptron(myperceptron,LR,input,output)
    
% function myperceptron=train_perceptron(myperceptron,LR,input,output)
% funcion que modifica los pesos del perceptron para que vaya aprendiendo
% a partir de los valores de entrada que se le indican
% ESTE PERCEPTRON UTILIZA:
 % Funcion sigma SIGMOIDAL
 % Entrenamiento DELTA RULE
% INPUTS:
 % myperceptron: estructura con el perceptron
 %myperceptron.bias: %Bias del perceptron (e.g. 1)
 %myperceptron.weights: %Pesos del perceptron
 % LR: learning rate (e.g. 0.7)
 % input: matriz con valores de entrada de entrenamiento (e.g. [E1 E2])
 % output: vector con valores de salida de entrenamiento (e.g. [SE])
% OUTPUT:
 % myperceptron: estructura con el perceptron ya entrenado
 %myperceptron.bias: %Bias del perceptron (e.g. 1)
 %myperceptron.weights: %Pesos del perceptron ya entrenado

 for i=1:5000
     v=myperceptron.weights(1)*myperceptron.bias+myperceptron.weights(2)*input(i,1)+myperceptron.weights(3)*input(i,2); % Cálculo de la salida
     y=1/(1+exp(-v)); % Función de activación
     err=output(i)-y;
     myperceptron.weights=myperceptron.weights+LR*err*[1;input(i,1);input(i,2)]; % Actualización de los pesos
 end
 
end

function output=useperceptron(myperceptron,input)
    
% function out=useperceptron(myperceptron,input)
% funcion que utiliza el perceptron para calcular las salidas a partir de
% las entradas de acuerdo con lo que haya aprendido el perceptron en la
% fase de entrenamiento

for i=1:20
     v=myperceptron.weights(1)*myperceptron.bias+myperceptron.weights(2)*input(i,1)+myperceptron.weights(3)*input(i,2); % Cálculo de la salida
     output(i)=1/(1+exp(-v)); % Función de activación
end
output=output(:);
end