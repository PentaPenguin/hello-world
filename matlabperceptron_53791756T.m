function matlabperceptron_53791756T
% Funcion principal que realiza las funciones de
%1) Creación de las variables para el banco de entrenamiento y banco de
 %validación
%2) Creación de la red neuronal de DOS CAPAS
%3) Entrenamiento de la red neuronal con el set de valores del banco de
 %entrenamiento
%4) Validación de la red con el banco de validación.
%5) Calculo y representación del error cometido.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Creamos entradas y salidas de entrenamiento para una funcion AND
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inicializamos las variables E1, E2 y SE ideales para entrenamiento
VT=1000; %numero de muestras de entrada
NV=round(VT); %aseguramos que VT sea un numero entero
E1=round(rand(NV,1)); %vector de NV valores de entrada en pin 1
E2=round(rand(NV,1)); %vector de NV valores de entrada en pin 2
SE=double(xor(E1,E2)); %vector salida ideal para las entradas E1 y E
%%%% Creamos entradas y salidas de validacion E1V, E2V, SEV para test
NVV=10;
E1V=round(rand(NVV,1));
E2V=round(rand(NVV,1));
SEV=double(xor(E1V,E2V));
% Inicializamos una red neuronal para 2 entradas %%%
mynn=feedforwardnet(2);
mynn=configure(mynn,[E1';E2'],SE');
% Entrenamos la red neuronal para un LR, por defecto 0.7
LR=0.7;
mynn.trainParam.lr = LR;
[mynn,tr]=train(mynn,[E1';E2'], SE');
%evaluamos la red
S_est=mynn([E1V';E2V']);
%Calculamos y representamos el error cometido
[SEV' S_est];
error=mean(abs(SEV'-S_est));
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
legend('Correct Values','Neural Network Output')
title('Evaluation of MULTILAYER Output for an XOR','FontWeight','bold')
end

function [mynn]=initialize_nn(n_inputs)
    
%function [myperceptron]=initialize_perceptron(n_inputs)
%funcion que inicializa un perceptron y le asigna pesos aleatorios
%funcion que inicializa una estructura de perceptron
% INPUTS:
 % n_inputs: numero de entradas al perceptron
% OUTPUTS:
 % myperceptron: estructura con el perceptron
 %myperceptron.bias: %Bias del perceptron (e.g. 1)
 %myperceptron.weights: %Pesos del perceptron (habrá tantos
 %como indice n_inputs +1 )
 
rand('state',sum(100*clock)); %inicializa random en función del reloj

mynn.bias=1;
mynn.weights=2*rand(n_inputs+1,3)-1;
wei=mynn.weights
end


function mynn=train_nn(mynn,LR,input,output)

% funcion que modifica los pesos la red para que vaya aprendiendo
% ESTE PERCEPTRON UTILIZA:
 % Funcion sigma SIGMOIDAL
 % Entrenamiento BACKPROPAGATION
% INPUTS:
 % mynn: estructura con el perceptron
 %mynnT.bias: %Bias del perceptron (e.g. 1)
 %mynnT.weights: %Pesos del perceptron
 % LR: learning rate (e.g. 0.7)
 % input: matriz con valores de entrada de entrenamiento (e.g. [E1 E2])
 % output: vector con valores de salida de entrenamiento (e.g. [SE])
% OUTPUT:
 % mynnT: estructura con el perceptron ya entrenado
 %mynnT.bias: %Bias del perceptron (e.g. 1)
 %mynnT.weights: %Pesos del perceptron ya entrenado
 
 [nsamples,ninputs]=size(input);

  for i=1:nsamples
      
     %Calculo y1 (primera neurona)
     h1=mynn.weights(1,1)*mynn.bias+mynn.weights(2,1)*input(i,1)+mynn.weights(3,1)*input(i,2);
     y1=1/(1+exp(-h1));
     %Calculo y2 (segunda neurona)
     h2=mynn.weights(1,2)*mynn.bias+mynn.weights(2,2)*input(i,1)+mynn.weights(3,2)*input(i,2);
     y2=1/(1+exp(-h2));
     %Calculo y3 (tercera neurona)
     h3=mynn.weights(1,3)*mynn.bias+mynn.weights(2,3)*y1+mynn.weights(3,3)*y2;
     y3=1/(1+exp(-h3));
     
     %Calculo error (tercera neurona)
     err3=y3*(1-y3)*(output(i)-y3);
     %Calculo error (primera neurona)
     err1=y1*(1-y1)*mynn.weights(2,3)*err3;
     %Calculo error (segunda neurona)
     err2=y2*(1-y2)*mynn.weights(3,3)*err3;
     
     %Actualizacion pesos
     mynn.weights(:,1)=mynn.weights(:,1)+LR*err1*[mynn.bias;input(i,1);input(i,2)];
     mynn.weights(:,2)=mynn.weights(:,2)+LR*err2*[mynn.bias;input(i,1);input(i,2)];
     mynn.weights(:,3)=mynn.weights(:,3)+LR*err3*[mynn.bias;y1;y2];
     
  end
  
end

function output=usenn(mynn,input)
% function out=useperceptron(myperceptron,input)
% funcion que utiliza el perceptron para calcular las salidas a partir de
% las entradas de acuerdo con lo que haya aprendido el perceptron en la
% fase de entrenamiento
    [nsamples,ninputs]=size(input);
 for i=1:nsamples
      
     %Calculo y1 (primera neurona)
     h1=mynn.weights(1,1)*mynn.bias+mynn.weights(2,1)*input(i,1)+mynn.weights(3,1)*input(i,2);
     y1=1/(1+exp(-h1));
     %Calculo y2 (segunda neurona)
     h2=mynn.weights(1,2)*mynn.bias+mynn.weights(2,2)*input(i,1)+mynn.weights(3,2)*input(i,2);
     y2=1/(1+exp(-h2));
     %Calculo y3 (tercera neurona)
     h3=mynn.weights(1,3)*mynn.bias+mynn.weights(2,3)*y1+mynn.weights(3,3)*y2;
     y3=1/(1+exp(-h3));
     output(i)=y3;
     
 end
 
 output=output(:);

end