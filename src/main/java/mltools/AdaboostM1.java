package mltools;

import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.AdditiveRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class AdaboostM1 {

    static AdaBoostM1 trainModel(String arffFile) throws Exception {

        File inputFile = new File(arffFile); //训练文件
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances insTrain = loader.getDataSet(); // 读入训练文件
        insTrain.setClassIndex(2); //设置分类属性所在列号（第一行为0号）
        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
        adaBoostM1.buildClassifier(insTrain);
        return adaBoostM1;
    }

    public List<Double> adaboostMain(){

        List<Double> clusterResult = new ArrayList<Double>();
        try{
            final String arffTestFilePath = "D:\\javaProjects\\ml\\logisticRegressionTestData.arff";
            final String arffTrainFilePath = "D:\\javaProjects\\ml\\logisticRegressionTrainData.arff";
            AdaBoostM1 adaBoostM1 = trainModel(arffTrainFilePath);
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(arffTestFilePath)); //测试文件
            Instances insTest =loader.getDataSet(); // 读入文件
            insTest.setClassIndex(2); //设置分类属性所在列号（第一行为0号）
            double sum = insTest.numInstances(); //测试实例数
            double right=0.0f;
            for(int i=0;i<sum;i++){
                Instance ins = insTest.instance(i);
                //将划分结果存储在list中
                clusterResult.add(adaBoostM1.classifyInstance(ins));
                if(adaBoostM1.classifyInstance(ins)==ins.classValue()) {
                    right++;
                    System.out.println("No.\t" + i + "\t" + ins.classValue() + " RIGHT");
                }
                else {
                    System.out.println("No.\t" + i + "\t" + ins.classValue() + " WRONG");
                }
            }
            // 打印出分类的精确度
            System.out.println("classification precision:" + (right/sum));
            return clusterResult;

        }catch (Exception e){
            //todo 异常处理
            e.printStackTrace();
            return clusterResult;
        }

    }

}
