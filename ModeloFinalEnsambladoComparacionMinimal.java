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
import weka.classifiers.meta.Vote;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.SelectedTag;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.Hashtable;
import java.util.Arrays;
import java.util.stream.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class ModeloFinalEnsambladoComparacionMinimal {
    public ModeloFinalEnsambladoComparacionMinimal() {
        super();
    }

    protected Classifier m_Classifier = null;
    protected String m_TrainingFile = null;
    protected Instances m_Training = null;
    protected Evaluation m_Evaluation = null;
    protected Integer key_fold = null;
    public static int classIndex = 20;

    public Evaluation execute() throws Exception {
        // train classifier on complete file for tree
        m_Classifier.buildClassifier(m_Training);
        
        // 10fold CV with seed=1~10
        m_Evaluation = new Evaluation(m_Training);
        m_Evaluation.crossValidateModel(
            m_Classifier, m_Training, 10, m_Training.getRandomNumberGenerator(key_fold));
        
        return m_Evaluation;
    }

    public void setTraining(String name) throws Exception {
        m_TrainingFile = name;
        m_Training     = new Instances(
                            new BufferedReader(new FileReader(m_TrainingFile)));
        m_Training.setClassIndex(m_Training.numAttributes() - 1);

    }

    public static String[] datasets = new String[]{
        "./4/data_filled_r.arff",
        "./4/data_filled_weka.arff"
    };

    public static String[] results = new String[]{
        "./results/4.final/final_model_comparison_minimal_stats_data_filled_r.csv",
        "./results/4.final/final_model_comparison_minimal_stats_data_filled_weka.csv"
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
        Evaluation eval;

        int classIndex = 20;

        Classifier[] algoritmos = new Classifier[] {
            new MultilayerPerceptron(),
            new NaiveBayes(),
            new RandomTree(),
            new RBFClassifier()
        };

        // Para cada dataset
        for (int j=0; j < datasets.length; j++) {
            ModeloFinalEnsambladoComparacionMinimal parseStats = new ModeloFinalEnsambladoComparacionMinimal();
            // Conjunto de datos a analizar
            parseStats.setTraining(datasets[j]);
            // Para cada clasificador
            try {
                PrintWriter pw = new PrintWriter(new File(results[j]));
                // Campos del archivo CSV
                pw.println(cabecera_csv);

                // Aleatoriedad de los subconjuntos para correr el 10-CV
                for (parseStats.key_fold = 1; parseStats.key_fold < 11; parseStats.key_fold++){
                    // Instanciar y Configurar el sistema ensamblado
                    Vote ensamblado = new Vote();
                    ensamblado.setClassifiers(algoritmos);
                    //ensamblado.finalizeAggregation();
                    // AVERAGE_RULE, PRODUCT_RULE, MAJORITY_VOTING_RULE, MIN_RULE, MAX_RULE, MEDIAN_RULE
                    ensamblado.setCombinationRule(new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES));
                    // Establecer la instancia como algoritmo clasificador
                    parseStats.m_Classifier = ensamblado;
                    // Corremos el algoritmo
                    eval = parseStats.execute();
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
            }
            
        }
    }
}