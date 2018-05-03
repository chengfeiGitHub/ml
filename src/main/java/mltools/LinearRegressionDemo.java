package mltools;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;


/**
 * 线性回归
 * 具体数据解释见数据文件
 */
public class LinearRegressionDemo {

    /**
     *
     * @param arffFile
     * @param classIndex  预测变量的索引，从0开始
     * @return
     * @throws Exception
     */
    public AbstractClassifier trainModel(String arffFile, int classIndex) throws Exception {

        File inputFile = new File(arffFile); //训练文件
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances insTrain = loader.getDataSet(); // 读入训练文件
        insTrain.setClassIndex(classIndex);

        LinearRegression linear = new LinearRegression();
        linear.buildClassifier(insTrain);//根据训练数据构造分类器
        int i =0 ;
        //依次打印出各个变量的系数，这个后期是否需要传给前端在讨论
        for (double coeff : linear.coefficients()){
            System.out.println("The " + i++ +"'th coefficient is:" + coeff);
        }
        return linear;
    }

    public void linearRegressionDemoMain() {

        final String arffTrainData = "D:\\javaProjects\\ml\\linearRegressionTrainData.arff";
        try{

            AbstractClassifier classifier = trainModel(arffTrainData, 5);

            Instance ins = new weka.core.SparseInstance(5);
           //设置测试数据，这些需要传入，根据前5个数据预测第6个，房价
            ins.setValue(0, 990.8);
            ins.setValue(1, 1080.8);
            ins.setValue(2, 3);
            ins.setValue(3, 0);
            ins.setValue(4, 1);

            double price = classifier.classifyInstance(ins);
            System.out.println("Price: " + price);

        }catch (Exception e){
            e.printStackTrace();
        }

    }

}
