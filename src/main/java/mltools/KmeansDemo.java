package mltools;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class KmeansDemo {
    public List<Integer> kmeansAlogorithm(){

       //这个结果集存储的是每个样本的所属类别，从0开始
        List<Integer> clusterResult  = new ArrayList<Integer>();

        try {
            // 读入样本数据
//            File file = new File("D:\\javaProjects\\ml\\kmeansIrisData.arff");
//            ArffLoader loader = new ArffLoader();
//            loader.setFile(file);
//            Instances ins = loader.getDataSet();


            DataSource source =new DataSource("D:\\javaProjects\\ml\\kmeansIrisData.csv");
            Instances ins = source.getDataSet();


            // 初始化聚类器 （加载算法）
            SimpleKMeans KM = new SimpleKMeans();
            KM.setNumClusters(3);       //设置聚类要得到的类别数量  这个后期需要传入
            KM.buildClusterer(ins);     //开始进行聚类

            for (int i = 0; i < ins.numInstances(); i++) {
                clusterResult.add(KM.clusterInstance(ins.instance(i)));
                //                                                       获取每个样本对应的类别index
                System.out.println( ins.instance(i) + " is in cluster: " + KM.clusterInstance(ins.instance(i)));
            }

            //获取聚类结果的误差平方和,这个是聚类评价指标
            double squaredError = KM.getSquaredError();
            System.out.println("误差平方和为: " + squaredError);

            //获取聚类结果中每一个类的个数
            double[] eachClusterCount = KM.getClusterSizes();

            return clusterResult;

        } catch(Exception e) {
            e.printStackTrace();
            //todo 异常处理
            return null;
        }
    }
}
