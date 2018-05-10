package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Dataset {
    private INDArray x;
    private INDArray y;

    public Dataset(final INDArray x, final INDArray y) {
        this.x = x;
        this.y = y;
    }

    public INDArray getX() {
        return x;
    }

    public INDArray getY() {
        return y;
    }
}
