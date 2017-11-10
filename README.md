# Neural Network of Bike Sharing

## Dataset

The data are from reference [1]. [link](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Data Exploration

The first 5 hous of data is shown as below:

<img src="https://filedn.com/lUE8ye7yWpzFOF1OFLVsPau/Github/bike_sharing/pics/data.png" alt="data" width="900"> 

The data (the number of the bikes rented) of the first 10 days is shown as below:

<img src="https://filedn.com/lUE8ye7yWpzFOF1OFLVsPau/Github/bike_sharing/pics/data_exp.png" alt="data_plot" width="600"> 

## Data Preprocessing

* Features `season`, `weathersit`, `mnth`, `hr`, `weekday` will be one-hot coded.

* Features `instant`, `dteday`, `atemp`, `workingday` will be removed. (The not one-hot coded features `season`, `weathersit`, `mnth`, `hr`, `weekday` will be also removed.)

* Featrues `casual`, `registered`, `cnt`, `temp`, `hum`, `windspeed` will be normalized, which will make the calculation in the neural network easier and more precise. The transformation information will be saved (mean value and variance), in order to transform the scaled prediction back to the origin value.

## Model Description (Fully Connected Neural Network)

```
[inputs] -> [hidden layer] -> (activation: sigmoid) -> [prediction]
```

This is a fully connected neural network.

* In the input layer `inputs`, the number of nodes is the same as the number of the features.

* In the hidden layer `hidden layer`, the number of nodes can be defined. (in the test: 15)

* The layer `prediction` there are three nodes, respectively `cnt`, `casual`, `registered`.

## Training

The training loss is shown as below:

<img src="https://filedn.com/lUE8ye7yWpzFOF1OFLVsPau/Github/bike_sharing/pics/losses.png" alt="data_plot" width="600"> 

From the diagram above, we can see that the model is a little bit overfitted.

## Prediction

The data used for prediction is the test_data. The result of prediction is shown as below:

<img src="https://filedn.com/lUE8ye7yWpzFOF1OFLVsPau/Github/bike_sharing/pics/prediction.png" alt="data_plot" width="900"> 

## References

1. Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3. [link](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

2. Udacity Deep Learning Nanodegree 

3. [Data Normalization (Wikipedia)](https://en.wikipedia.org/wiki/Normalization_(statistics))

4. [One-hot (Wikipedia)](https://en.wikipedia.org/wiki/One-hot)