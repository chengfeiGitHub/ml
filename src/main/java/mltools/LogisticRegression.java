package mltools;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;

public class LogisticRegression {

    static Logistic trainModel(String arffFile) throws Exception {

        File inputFile = new File(arffFile); //训练文件
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances insTrain = loader.getDataSet(); // 读入训练文件

//        ConverterUtils.DataSource source =new ConverterUtils.DataSource(arffFile);
//        Instances insTrain = source.getDataSet();
        insTrain.setClassIndex(2); //设置分类属性所在列号（第一行为0号）
        Logistic logic=new Logistic();

        logic.buildClassifier(insTrain);//根据训练数据构造分类器

        return logic;
    }

    public List<Double> logistRegressionMain(){

        List<Double> clusterResult = new ArrayList<Double>();
        try{
            final String arffTestFilePath = "D:\\javaProjects\\ml\\test.arff";
            final String arffTrainFilePath = "D:\\javaProjects\\ml\\train.arff";

            Logistic logic = trainModel(arffTrainFilePath);


            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(arffTestFilePath)); //测试文件
            Instances insTest =loader.getDataSet(); // 读入文件
//            ConverterUtils.DataSource source =new ConverterUtils.DataSource(arffTestFilePath);
//            Instances insTest = source.getDataSet();
            insTest.setClassIndex(2); //设置分类属性所在列号（第一行为0号）

            double sum = insTest.numInstances(); //测试实例数
            double right=0.0f;
            for(int i=0;i<sum;i++){

                Instance ins = insTest.instance(i);
                //将划分结果存储在list中
                clusterResult.add(logic.classifyInstance(ins));

                if(logic.classifyInstance(ins)==ins.classValue()) {

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
