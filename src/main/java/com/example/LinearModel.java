package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LinearModel {
    private final Logger logger = LoggerFactory.getLogger(LinearModel.class);
    private INDArray theta;

    public LinearModel(final INDArray theta) {
        this.theta = theta;
        logger.debug("Model Parameters = {}", theta);
    }

    public double predict(final INDArray features) {
        INDArray ones = Nd4j.ones(features.rows(), 1);
        INDArray x = Nd4j.hstack(ones, features);
        return x.mmul(theta).getDouble(0, 0);
    }

    public INDArray getParameters() {
        return theta;
    }
}
