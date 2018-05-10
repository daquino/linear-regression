package com.example;

import com.opencsv.CSVReader;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class LinearRegressionTest {

    private final Logger logger = LoggerFactory.getLogger(LinearRegressionTest.class);

    /**
     * For this test, the training set consists of [2, 3] with target values (y) of [4, 6].
     * It should be pretty obvious that the line that best fits is represented by the equation
     * y = 2x.  This means that the model built should have [0, 2] is its parameters (thetas).
     */
    @Test
    public void testSimpleLinearRegression() {
        INDArray trainingData = Nd4j.create(new double[][]{{2}, {3}});
        INDArray targets = Nd4j.create(new double[][]{{4}, {6}});
        INDArray features = Nd4j.create(new double[]{3.5});

        System.out.println(trainingData.shapeInfoToString());


        LinearModel model = LinearSolver.solveIterable(trainingData, targets, 0.01, 20_000);
        double prediction = model.predict(features);

        Assert.assertEquals(7.0, prediction, 0.1);
        logger.debug("Prediction = " + prediction);
    }

    /**
     * Same test as above but the version that uses a threshold of 0.0000000001 to stop iterating through gradient descent.
     */
    @Test
    public void testSimpleThresholdLinearRegression() {
        //given
        INDArray trainingData = Nd4j.create(new double[][]{{1}, {2}, {3}});
        INDArray targets = Nd4j.create(new double[][]{{2}, {4}, {6}});
        INDArray features = Nd4j.create(new double[]{4});

        //when
        LinearModel model = LinearSolver.solveThreshold(trainingData, targets, 0.01, 1.0e-9);
        double prediction = model.predict(features);

        //then
        Assert.assertEquals(8.0, prediction, 0.1);
        logger.debug("Prediction = " + prediction);
    }

    /**
     * Linear regression with housing price data with 1000 iterations of gradient descent.
     */
    @Test
    public void testHousingLinearRegression() {
        Dataset data = read("src/test/resources/updated_training.csv");
        INDArray features = Nd4j.create(new double[]{3343, 8});
        double expectedPrediction = 250_000;

        LinearModel model = LinearSolver.solveIterable(data.getX(), data.getY(), 1.0e-8, 100_000);
        double prediction = model.predict(features);

        Assert.assertEquals(expectedPrediction, prediction, expectedPrediction * 0.075);
        logger.debug("Prediction = " + prediction);
    }

    public Dataset read(final String path) {
        List<double[]> trainingSet = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            String[] line;
            reader.readNext();
            while ((line = reader.readNext()) != null) {
                double[] trainingRecord = new double[line.length - 1];
                double[] targetRecord = new double[1];
                for (int i = 0; i < line.length; i++) {
                    if (i == line.length - 1) {
                        targetRecord[0] = Double.valueOf(line[i]);
                    }
                    else {
                        trainingRecord[i] = Double.valueOf(line[i]);
                    }
                }
                trainingSet.add(trainingRecord);
                targets.add(targetRecord);
            }
            return new Dataset(Nd4j.create(trainingSet.toArray(new double[trainingSet.size()][])),
                    Nd4j.create(targets.toArray(new double[targets.size()][])));
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    /**
     * Linear regression with housing price data with threshold of 0.0000000001.
     */
    @Test
    public void testHousingThresholdLinearRegression() {
        Dataset data = read("src/test/resources/updated_training.csv");
        INDArray features = Nd4j.create(new double[]{3343, 8});
        double expectedPrediction = 250_000;

        LinearModel model = LinearSolver.solveThreshold(data.getX(), data.getY(), 1.0e-8, 1.0e-9);
        double prediction = model.predict(features);

        Assert.assertEquals(expectedPrediction, prediction, expectedPrediction * 0.075);
        logger.debug("Prediction = " + prediction);
    }
}
