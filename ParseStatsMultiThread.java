import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.functions.RBFClassifier;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE;
import weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Hashtable;
import java.util.Arrays;
import java.util.stream.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class ParseStatsMultiThread {
    public ParseStatsMultiThread() {
        super();
    }

    protected Classifier m_Classifier = null;
    protected String m_TrainingFile = null;
    protected Instances m_Training = null;
    protected Evaluation m_Evaluation = null;
    protected Integer key_fold = null;

    public void setTraining(String name) throws Exception {
        m_TrainingFile = name;
        m_Training     = new Instances(
                            new BufferedReader(new FileReader(m_TrainingFile)));
        m_Training.setClassIndex(m_Training.numAttributes() - 1);

    }

    public Evaluation execute() throws Exception {
        // train classifier on complete file for tree
        m_Classifier.buildClassifier(m_Training);
        
        // 10fold CV with seed=1~10
        m_Evaluation = new Evaluation(m_Training);
        m_Evaluation.crossValidateModel(
            m_Classifier, m_Training, 10, m_Training.getRandomNumberGenerator(key_fold));
        
        return m_Evaluation;
    }

    public static Double promediarArray(Double[] array) {
        return Arrays.stream(array).mapToDouble(Double::doubleValue).sum() / array.length;
    }

    public static int classIndex = 20;
    public static String data_path = "./4/";
    public static String result_path = "./results/4/";

    public static String[] datasets = new String[]{
        "data_filled_r",
        "data_filled_weka"
    };

    public static String cabecera_csv = 
        "areaUnderPRC,areaUnderROC,avgCost,correct,coverageOfTestCasesByPredictedRegions," +
        "errorRate,falseNegativeRate,falsePositiveRate,fMeasure,incorrect,kappa,KBInformation," +
        "KBMeanInformation,meanAbsoluteError,numFalseNegatives,numFalsePositives,numInstances," +
        "numTrueNegatives,numTruePositives,pctCorrect,pctIncorrect,pctUnclassified,precision," +
        "recall,relativeAbsoluteError,rootMeanSquaredError,totalCost,trueNegativeRate," +
        "truePositiveRate,unclassified,weightedAreaUnderPRC,weightedAreaUnderROC," +
        "weightedFalseNegativeRate,weightedFalsePositiveRate,weightedFMeasure," +
        "weightedRecall,weightedTrueNegativeRate,weightedTruePositiveRate,key_fold";
    
    public static StringBuilder buildString(Evaluation eval, StringBuilder sb) throws Exception {
        try {
            sb.append(eval.areaUnderPRC(classIndex));
        } catch (NullPointerException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        try {
            sb.append(eval.areaUnderROC(classIndex));
        } catch (NullPointerException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        sb.append(eval.avgCost());
        sb.append(',');
        sb.append(eval.correct());
        sb.append(',');
        sb.append(eval.coverageOfTestCasesByPredictedRegions());
        sb.append(',');
        sb.append(eval.errorRate());
        sb.append(',');
        sb.append(eval.falseNegativeRate(classIndex));
        sb.append(',');
        sb.append(eval.falsePositiveRate(classIndex));
        sb.append(',');
        try {
            sb.append(eval.fMeasure(classIndex));
        } catch (ArrayIndexOutOfBoundsException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        sb.append(eval.incorrect());
        sb.append(',');
        sb.append(eval.kappa());
        sb.append(',');
        sb.append(eval.KBInformation());
        sb.append(',');
        sb.append(eval.KBMeanInformation());
        sb.append(',');
        sb.append(eval.meanAbsoluteError());
        sb.append(',');
        sb.append(eval.numFalseNegatives(classIndex));
        sb.append(',');
        sb.append(eval.numFalsePositives(classIndex));
        sb.append(',');
        sb.append(eval.numInstances());
        sb.append(',');
        sb.append(eval.numTrueNegatives(classIndex));
        sb.append(',');
        sb.append(eval.numTruePositives(classIndex));
        sb.append(',');
        sb.append(eval.pctCorrect());
        sb.append(',');
        sb.append(eval.pctIncorrect());
        sb.append(',');
        sb.append(eval.pctUnclassified());
        sb.append(',');
        try {
            sb.append(eval.precision(classIndex));
        } catch (ArrayIndexOutOfBoundsException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        try {
            sb.append(eval.recall(classIndex));
        } catch (ArrayIndexOutOfBoundsException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        sb.append(eval.relativeAbsoluteError());
        sb.append(',');
        sb.append(eval.rootMeanSquaredError());
        sb.append(',');
        sb.append(eval.totalCost());
        sb.append(',');
        try {
            sb.append(eval.trueNegativeRate(classIndex));
        } catch (ArrayIndexOutOfBoundsException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        try {
            sb.append(eval.truePositiveRate(classIndex));
        } catch (ArrayIndexOutOfBoundsException e) {
            sb.append(Double.NaN);
            e.printStackTrace();
        };
        sb.append(',');
        sb.append(eval.unclassified());
        sb.append(',');
        sb.append(eval.weightedAreaUnderPRC());
        sb.append(',');
        sb.append(eval.weightedAreaUnderROC());
        sb.append(',');
        sb.append(eval.weightedFalseNegativeRate());
        sb.append(',');
        sb.append(eval.weightedFalsePositiveRate());
        sb.append(',');
        sb.append(eval.weightedFMeasure());
        sb.append(',');
        sb.append(eval.weightedRecall());
        sb.append(',');
        sb.append(eval.weightedTrueNegativeRate());
        sb.append(',');
        sb.append(eval.weightedTruePositiveRate());
        return sb;
    }

    public static void main(String[] args) throws Exception {
        Thread[] hilos = new Thread[]{
            new Thread("MultilayerPerceptron") {
                public void run(){
                    Classifier algoritmo = new MultilayerPerceptron();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("RBFNetwork") {
                public void run(){
                    Classifier algoritmo = new RBFNetwork();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("RBFClassifier") {
                public void run(){
                    Classifier algoritmo = new RBFClassifier();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("SimpleLogistic") {
                public void run(){
                    Classifier algoritmo = new SimpleLogistic();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("Logistic") {
                public void run(){
                    Classifier algoritmo = new Logistic();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("SMO") {
                public void run(){
                    Classifier algoritmo = new SMO();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("RandomForest") {
                public void run(){
                    Classifier algoritmo = new RandomForest();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("NaiveBayes") {
                public void run(){
                    Classifier algoritmo = new NaiveBayes();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("A1DE") {
                public void run(){
                    Classifier algoritmo = new A1DE();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("A2DE") {
                public void run(){
                    Classifier algoritmo = new A2DE();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("BayesNet") {
                public void run(){
                    Classifier algoritmo = new BayesNet();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("RandomTree") {
                public void run(){
                    Classifier algoritmo = new RandomTree();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("REPTree") {
                public void run(){
                    Classifier algoritmo = new REPTree();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("J48") {
                public void run(){
                    Classifier algoritmo = new J48();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("IBk1") {
                public void run(){
                    Classifier algoritmo = new IBk(1);
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("IBk5") {
                public void run(){
                    Classifier algoritmo = new IBk(5);
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("IBk10") {
                public void run(){
                    Classifier algoritmo = new IBk(10);
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            },
            new Thread("DecisionTable") {
                public void run(){
                    Classifier algoritmo = new DecisionTable();
                    for (String d : datasets) {
                        ParseStatsMultiThread parseStats = new ParseStatsMultiThread();
                        // Build the data path
                        try {
                            parseStats.setTraining(data_path + d + ".arff");
                            PrintWriter pw = new PrintWriter(new File(result_path + d + getName() + ".csv"));
                            // Campos del archivo CSV
                            pw.println(cabecera_csv);
                            // Aleatoriedad de los subconjuntos para correr el 10-CV
                            for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                                // Selección de algoritmo clasificador
                                parseStats.m_Classifier = algoritmo;
                                // Corremos el algoritmo
                                Evaluation eval = parseStats.execute();
                                // Sección de recolección de métricas para cada corrida de 10-CV
                                StringBuilder sb = new StringBuilder();
                                sb = buildString(eval, sb);
                                sb.append(',');
                                sb.append(parseStats.key_fold);
                                pw.println(sb.toString());
                                pw.flush();
                            }
                            pw.close();
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        };
        
        for (Thread t : hilos) {
            t.start();
        }
    }
}