package Modelo;

import Controle.Comunicador;
import Recursos.Arquivo;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 *
 * @author Jônatas Trabuco Belotti [jonatas.t.belotti@hotmail.com]
 */
public class MLP {

  public static final int NUM_ENTRADAS = 15;
  private final int NUM_NEU_CAMADA_ESCONDIDA = 25;
  public static final int NUM_NEU_CAMADA_SAIDA = 1;
  private final double TAXA_APRENDIZAGEM = 0.1;
  private final double PRECISAO = 0.5 * Math.pow(10D, -6D);
  private final double FATOR_MOMENTUM = 0.8;
  private final double BETA = 1.0;
  private final int LIMITE_NUM_EPOCAS = 50000;

  private int numEpocas;
  private double[] entradas;
  private double[][] pesosCamadaEscondida;
  private double[][] pesosCamadaEscondidaProximo;
  private double[][] pesosCamadaEscondidaAnterior;
  private double[][] pesosCamadaSaida;
  private double[][] pesosCamadaSaidaProximo;
  private double[][] pesosCamadaSaidaAnterior;
  private double[] potencialCamadaEscondida;
  private double[] saidaCamadaEscondida;
  private double[] potencialCamadaSaida;
  private double[] saidaCamadaSaida;
  private double[] saidaEsperada;
  private double[] gradienteCamadaSaida;
  private double[] gradienteCamadaEscondida;

  public MLP() {
    Random random;

    entradas = new double[NUM_ENTRADAS + 1];
    pesosCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaEscondidaAnterior = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaEscondidaProximo = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    pesosCamadaSaidaAnterior = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    pesosCamadaSaidaProximo = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    potencialCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA + 1];
    saidaCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA + 1];
    potencialCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    saidaCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    saidaEsperada = new double[NUM_NEU_CAMADA_SAIDA];
    gradienteCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    gradienteCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA];

    //Iniciando pesos sinapticos
    random = new Random();

    for (int i = 0; i < NUM_NEU_CAMADA_ESCONDIDA; i++) {
      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        pesosCamadaEscondida[i][j] = random.nextDouble();
      }
    }

    for (int i = 0; i < NUM_NEU_CAMADA_SAIDA; i++) {
      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        pesosCamadaSaida[i][j] = random.nextDouble();
      }
    }
  }

  public boolean treinar(Arquivo arquivoTreinamento) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    double erroAtual;
    double erroAnterior;
    long tempInicial;

    copiarMatriz(pesosCamadaEscondida, pesosCamadaEscondidaAnterior);
    copiarMatriz(pesosCamadaSaida, pesosCamadaSaidaAnterior);

    tempInicial = System.currentTimeMillis();
    numEpocas = 0;
    erroAtual = erroQuadraticoMedio(arquivoTreinamento);

    Comunicador.iniciarLog("Início treinamento da MLP");
    Comunicador.addLog(String.format("Erro inicial: %.6f", erroAtual).replace(".", ","));
    imprimirPesos();
    Comunicador.addLog("Época Eqm");

    try {
      do {
        this.numEpocas++;
        erroAnterior = erroAtual;
        arq = new FileReader(arquivoTreinamento.getCaminhoCompleto());
        lerArq = new BufferedReader(arq);

        linha = lerArq.readLine();
        if (linha.contains("x") || linha.contains("f")) {
          linha = lerArq.readLine();
        }

        //Iniciando janela de entradas
        entradas[0] = -1D;
        for (int i = entradas.length - 1; i > 0; i--) {
          entradas[i] = separarEntrada(linha);
          linha = lerArq.readLine();
        }

        while (linha != null) {
          saidaEsperada[0] = separarEntrada(linha);//Proxima entrada é a saída experada

          calcularSaidas();

          ajustarPesos();

          linha = lerArq.readLine();
          ajustarJanela();
          entradas[1] = saidaEsperada[0];
        }

        arq.close();
        erroAtual = erroQuadraticoMedio(arquivoTreinamento);
        Comunicador.addLog(String.format("%d   %.6f", numEpocas, erroAtual).replace(".", ","));
      } while (Math.abs(erroAtual - erroAnterior) > PRECISAO && numEpocas < LIMITE_NUM_EPOCAS);

      Comunicador.addLog(String.format("Fim do treinamento. (%.2fs)", (double) (System.currentTimeMillis() - tempInicial) / 1000D));
      imprimirPesos();
    } catch (FileNotFoundException ex) {
      return false;
    } catch (IOException ex) {
      return false;
    }

    return true;
  }

  public void testar(Arquivo arquivoTreinamento, Arquivo arquivoTeste) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    int numAmostras;
    double erroMedio;
    double variancia;
    double erro;

    numAmostras = 0;
    erroMedio = 0D;
    variancia = 0D;

    Comunicador.iniciarLog("Início teste da MLP");
    Comunicador.addLog("x  -- y");

    try {
      //Iniciando janela de entradas
      arq = new FileReader(arquivoTreinamento.getCaminhoCompleto());
      lerArq = new BufferedReader(arq);

      linha = lerArq.readLine();
      if (linha.contains("x") || linha.contains("f")) {
        linha = lerArq.readLine();
      }

      entradas[0] = -1D;
      for (int i = entradas.length - 1; i > 0; i--) {
        entradas[i] = separarEntrada(linha);
        linha = lerArq.readLine();
      }

      while (linha != null) {
        ajustarJanela();
        entradas[1] = separarEntrada(linha);
        linha = lerArq.readLine();
      }

      //Abrindo arquivos com dados de teste
      arq = new FileReader(arquivoTeste.getCaminhoCompleto());
      lerArq = new BufferedReader(arq);

      linha = lerArq.readLine();
      if (linha.contains("x") || linha.contains("f")) {
        linha = lerArq.readLine();
      }

      while (linha != null) {
        saidaEsperada[0] = separarEntrada(linha);//Proxima entrada é a saída experada

        numAmostras++;

        calcularSaidas();

        erro = (100D / saidaEsperada[0]) * Math.abs(saidaEsperada[0] - saidaCamadaSaida[0]);
        erroMedio += erro;
        variancia += Math.pow(erro, 2D);
        Comunicador.addLog(String.format("%.6f %.6f %.2f%%", saidaEsperada[0], saidaCamadaSaida[0], erro));

        ajustarJanela();
        entradas[1] = saidaCamadaSaida[0];
        linha = lerArq.readLine();
      }

      //Calculando erro relativo médio
      erroMedio = erroMedio / (double) numAmostras;

      //Calculando variância
      variancia = variancia - ((double) numAmostras * Math.pow(erroMedio, 2D));
      variancia = variancia / ((double) (numAmostras - 1));

      Comunicador.addLog("Fim do teste");
      Comunicador.addLog(String.format("Erro relativo médio: %.6f%%", erroMedio));
      Comunicador.addLog(String.format("Variância: %.6f%%", variancia));

      arq.close();
    } catch (FileNotFoundException ex) {
    } catch (IOException ex) {
    }
  }

  private double erroQuadraticoMedio(Arquivo arquivo) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    int numAmostras;
    double erroMedio;
    double erro;

    erroMedio = 0D;
    numAmostras = 0;

    try {
      arq = new FileReader(arquivo.getCaminhoCompleto());
      lerArq = new BufferedReader(arq);

      linha = lerArq.readLine();
      if (linha.contains("x") || linha.contains("f")) {
        linha = lerArq.readLine();
      }

      //Iniciando janela de entradas
      entradas[0] = -1D;
      for (int i = entradas.length - 1; i > 0; i--) {
        entradas[i] = separarEntrada(linha);
        linha = lerArq.readLine();
      }

      while (linha != null) {
        numAmostras++;
        saidaEsperada[0] = separarEntrada(linha);//Proxima entrada é a saída experada

        calcularSaidas();

        //Calculando erro
        erro = 0D;
        for (int i = 0; i < saidaCamadaSaida.length; i++) {
          erro = erro + Math.abs(saidaEsperada[i] - saidaCamadaSaida[i]);
        }
        erroMedio = erroMedio + erro;

        ajustarJanela();
        entradas[1] = saidaEsperada[0];
        linha = lerArq.readLine();
      }

      arq.close();
      erroMedio = erroMedio / (double) numAmostras;

    } catch (FileNotFoundException ex) {
    } catch (IOException ex) {
    }

    return erroMedio;
  }

  private double separarEntrada(String linha) {
    String[] vetor;
    int i;

    vetor = linha.split("\\s+");
    i = 0;

    if (vetor[0].equals("")) {
      i = 1;
    }

    return Double.parseDouble(vetor[i].replace(",", "."));
  }

  private void ajustarJanela() {
    for (int i = entradas.length - 1; i > 1; i--) {
      entradas[i] = entradas[i - 1];
    }
  }

  private void calcularSaidas() {
    double valorParcial;

    //Calculando saidas da camada escondida
    saidaCamadaEscondida[0] = -1D;
    potencialCamadaEscondida[0] = -1D;

    for (int i = 1; i < saidaCamadaEscondida.length; i++) {
      valorParcial = 0D;

      for (int j = 0; j < entradas.length; j++) {
        valorParcial += entradas[j] * pesosCamadaEscondida[i - 1][j];
      }

      potencialCamadaEscondida[i] = valorParcial;
      saidaCamadaEscondida[i] = funcaoLogistica(valorParcial);
    }

    //Calculando saida da camada de saída
    for (int i = 0; i < saidaCamadaSaida.length; i++) {
      valorParcial = 0D;

      for (int j = 0; j < saidaCamadaEscondida.length; j++) {
        valorParcial += saidaCamadaEscondida[j] * pesosCamadaSaida[i][j];
      }

      potencialCamadaSaida[i] = valorParcial;
      saidaCamadaSaida[i] = funcaoLogistica(valorParcial);
    }
  }

  private void ajustarPesos() {
    //Ajustando pesos sinapticos da camada de saida
    for (int i = 0; i < gradienteCamadaSaida.length; i++) {
      gradienteCamadaSaida[i] = (saidaEsperada[i] - saidaCamadaSaida[i]) * funcaoLogisticaDerivada(potencialCamadaSaida[i]);

      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        pesosCamadaSaidaProximo[i][j] = pesosCamadaSaida[i][j]
                + (FATOR_MOMENTUM * (pesosCamadaSaida[i][j] - pesosCamadaSaidaAnterior[i][j]))
                + (TAXA_APRENDIZAGEM * gradienteCamadaSaida[i] * saidaCamadaEscondida[j]);
      }
    }

    //Ajustando pesos sinapticos da camada escondida
    for (int i = 0; i < gradienteCamadaEscondida.length; i++) {
      gradienteCamadaEscondida[i] = 0D;
      for (int j = 0; j < NUM_NEU_CAMADA_SAIDA; j++) {
        gradienteCamadaEscondida[i] += gradienteCamadaSaida[j] * pesosCamadaSaida[j][i + 1];
      }
      gradienteCamadaEscondida[i] *= funcaoLogisticaDerivada(potencialCamadaEscondida[i + 1]);

      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        pesosCamadaEscondidaProximo[i][j] = pesosCamadaEscondida[i][j]
                + (FATOR_MOMENTUM * (pesosCamadaEscondida[i][j] - pesosCamadaEscondidaAnterior[i][j]))
                + (TAXA_APRENDIZAGEM * gradienteCamadaEscondida[i] * entradas[j]);
      }
    }

    //Copiando pesos
    copiarMatriz(pesosCamadaEscondida, pesosCamadaEscondidaAnterior);
    copiarMatriz(pesosCamadaEscondidaProximo, pesosCamadaEscondida);
    copiarMatriz(pesosCamadaSaida, pesosCamadaSaidaAnterior);
    copiarMatriz(pesosCamadaSaidaProximo, pesosCamadaSaida);
  }

  private void copiarMatriz(double[][] origem, double[][] destino) {
    for (int i = 0; i < origem.length; i++) {
      for (int j = 0; j < origem[i].length; j++) {
        if (destino.length > i) {
          if (destino[i].length > j) {
            destino[i][j] = origem[i][j];
          }
        }
      }
    }
  }

  private double funcaoLogistica(double valor) {
    return 1D / (1D + Math.pow(Math.E, -1D * BETA * valor));
  }

  private double funcaoLogisticaDerivada(double valor) {
    return (BETA * Math.pow(Math.E, -1D * BETA * valor)) / Math.pow((Math.pow(Math.E, -1D * BETA * valor) + 1D), 2D);
  }

  private void imprimirPesos() {
    String log;

    Comunicador.addLog("Pesos camada escondida:");

    for (int i = 0; i < NUM_NEU_CAMADA_ESCONDIDA; i++) {
      log = "N" + (i + 1) + " =";

      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        log += String.format(" %f", pesosCamadaEscondida[i][j]);
      }

      Comunicador.addLog(log);
    }

    Comunicador.addLog("Pesos camada de saída:");
    for (int i = 0; i < NUM_NEU_CAMADA_SAIDA; i++) {
      log = "N" + (i + 1) + " =";

      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        log += String.format(" %f", pesosCamadaSaida[i][j]);
      }

      Comunicador.addLog(log);
    }
  }

}
