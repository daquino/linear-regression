package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LinearSolver {
    final static Logger logger = LoggerFactory.getLogger(LinearSolver.class);

    /**
     * Perform linear regression iterating with a specified number of updates for gradient descent.
     *
     * @param trainingSet training inputs
     * @param y           training outputs
     * @param alpha       learning rate
     * @param iterations  number of iterations for gradient descent
     * @return a linear model fit to the training data
     */
    public static LinearModel solveIterable(final INDArray trainingSet, final INDArray y, final double alpha, int iterations) {
        int logThreshold = iterations / 10;

        //append 1 to each training example
        INDArray ones = Nd4j.ones(trainingSet.rows(), 1);
        INDArray x = Nd4j.hstack(ones, trainingSet);

        //start at 0s for model parameters
        INDArray theta = Nd4j.zeros(x.columns(), 1);

        //iterate 10,000 times to update thetas
        for (int i = 0; i < iterations; i++) {
            INDArray difference = calculateChange(x, y, alpha, theta);
            theta.subi(difference);
            if (i % logThreshold == 0 && logger.isDebugEnabled()) {
                logger.debug("Iteration {}: Parameters = {}, Cost = {}", i + 1, theta, calculateCost(theta, x, y));
            }
        }
        logger.debug("Performed {} iterations of gradient descent. Final cost {}", iterations, calculateCost(theta, x, y));

        return new LinearModel(theta);
    }

    /**
     * Calculates the amount to change model parameters for minimization.
     *
     * @param x     training inputs
     * @param y     training outputs
     * @param alpha learning rate
     * @param theta current model parameters
     * @return the amount to change model parameters
     */
    private static INDArray calculateChange(final INDArray x, final INDArray y,
                                            final double alpha, final INDArray theta) {
        //number of training samples
        double m = y.length();

        //prediction with current model parameters
        INDArray h = x.mmul(theta);

        //calculate errors
        INDArray errors = h.sub(y);

        //calculate derivative
        INDArray product = x.transpose().mmul(errors);
        INDArray derivative = product.mul(1 / m);

        // multiply by learning rate
        return derivative.mul(alpha);
    }

    /**
     * Calculates cost with specified model parameters using mean square error variant.
     *
     * @param theta current model parameters
     * @param x     training inputs
     * @param y     training outputs
     * @return cost with specified model parameters with MSE variant
     */
    public static double calculateCost(final INDArray theta, final INDArray x, final INDArray y) {
        //fetch the number of training samples
        double m = y.length();

        //calculate predictions with current parameters using all training inputs
        INDArray h = x.mmul(theta);

        // calculate errors from correct values
        INDArray errors = h.sub(y);

        // sum of the square the errors
        double errorsSquaredSum = errors.transpose().mmul(errors).getDouble(0, 0);

        // calculate the mean (variant) of the squared sum
        return ( 1 / ( 2 * m)) * errorsSquaredSum;
    }

    /**
     * Performs linear regression using a threshold to indicate a stopping point for gradient descent.
     *
     * @param trainingSet training inputs
     * @param y           training outputs
     * @param alpha       learning rate
     * @param threshold   cost function change threshold
     * @return a linear model fit to the training data
     */
    public static LinearModel solveThreshold(INDArray trainingSet, INDArray y, double alpha, double threshold) {
        int m = y.length();

        //append 1 to each training example
        INDArray ones = Nd4j.ones(trainingSet.rows(), 1);
        INDArray x = Nd4j.hstack(ones, trainingSet);

        //initialize theta starting point to [0,0]
        INDArray theta = Nd4j.zeros(x.columns(), 1);

        //perform iterations of gradient descent until reaching the specified cost function threshold
        double prevCost;
        double updateCost;
        int iterations = 0;
        do {
            prevCost = calculateCost(theta, x, y);
            INDArray difference = calculateChange(x, y, alpha, theta);
            theta.subi(difference);
            updateCost = calculateCost(theta, x, y);
            if (iterations % 100 == 0 && logger.isDebugEnabled()) {
                logger.debug("Iteration {}: Parameters = {}, Cost = {}", iterations + 1, theta, updateCost);
            }
            iterations++;
        } while (prevCost - updateCost > threshold);
        logger.debug("Performed {} iterations of gradient descent. Final cost {}", iterations, updateCost);
        return new LinearModel(theta);
    }
}
