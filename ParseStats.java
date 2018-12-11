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

public class ParseStats {
    public ParseStats() {
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

    String[] datasets = new String[]{
        "/home/gosorio/Projects/qoe_datasets/1/data_filled_r.arff",
        "/home/gosorio/Projects/qoe_datasets/1/data_filled_weka.arff"
    };

    public static void main(String[] args) throws Exception {
        Evaluation eval;

        Classifier[] algoritmos = new Classifier[]{
            new MultilayerPerceptron(),
            new RBFNetwork(),
            new RBFClassifier(),
            new SimpleLogistic(),
            new Logistic(),
            new SMO(),
            new RandomForest(),
            new NaiveBayes(),
            new A1DE(),
            new A2DE(),
            new BayesNet(),
            new RandomTree(),
            new REPTree(),
            new J48(),
            new IBk(1),
            new IBk(5),
            new IBk(10),
            new DecisionTable()
        };

        // Para cada dataset
        for (int j=0; j < datasets.length; j++){
            ParseStats parseStats = new ParseStats();
            
            // Conjunto de datos a analizar
            parseStats.setTraining(datasets[j]);

            // Para cada clasificador
            for (int i=0; i < algoritmos.length; i++){
                // Aleatoriedad de los subconjuntos para correr el 10-CV
                for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                    // Selección de algoritmo clasificador
                    parseStats.m_Classifier = algoritmos[i];
                    // Corremos el algoritmo
                    eval = parseStats.execute();
                    // Sección de promedios para cada corrida de 10-CV
                }
            }
        }
    }
}

/**
// Test
                    Double[] array = new Double[]{0.559322033898305,0.4915254237288136,0.5084745762711864};
                    System.out.println("Promedio de array con stream: " + promediarArray(array));
                    
                    System.out.println(parseStats.toString());
 */