# TensorFlow Boost
**TensorFlow Boost's Repository**

| **`Windows CPU`** | **`Windows GPU`** |
|-------------------|-------------------|
|<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Build_Passing.png" height="20" width="90"> | <img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Build_Passing.png" height="20" width="90">

Through project, we are creating a framework thanks to which you can read any kind of tag labeled data (like Kaggle problems, CSV and images); create train, validation and test set from them; choose the best machine learning algorithm for your data, and, besides, change the algorithm features.

Update Version 0.1:

  - Now it is possible to save models configurations by problem and load previous models configurations.
  - See json information by time in two different ways: "Information" and "Configuration".
  - See graphs progress during training. After each epoch it will be saved a graph. 
  - You can decide if you want to save graphs after validation/test accuracies are surpass your limit.
  - You can reset the configuration making backups.  
  - You can change dropout during training easily.
  - You can restore previous tensorflow models easily.
  - You have a method by problem: for each problem you can solve, you could created a method to process each input in a different way.
  - Yoy have a "Setting.json" file for each problem where you only have to put the paths where you want to process your problem.
  - You can see loss and accuracies in graphs and printed in the console.
  - You can easily change the epochs and batch sizes.
  - You have a **CNN example** treating a signal problem.
  
Next Version:

  - You will be able to do all this with a simple and beautiful user interface (in curse).
  - You will have an example of **a LSTM project** (in curse)
  
Future Versions:

  - In future, this project contains a graph visualization BEFORE TensorFlow generates his graphs.


All project use "Google Python Style Guide":
https://google.github.io/styleguide/pyguide.html

**TensorFlow Boost Web example (NEW):**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/tfboost-framework-web-example.png"><br>
<br>
</div>

**TensorFlow Boost works as follows:**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/CSV_Diagram.png"><br><br>
</div>

**An example of 'information.json':**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Information_Example.png"><br><br>
</div>

**An example of a Accuracy Graph:**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Graph_Accuracy.png"><br><br>
</div>

**An example of a Loss Graph:**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Graph_Loss.png"><br><br>
</div>

**An example of code: Step by step structure (Python-Tensorflow Code)**

<div align="center">
<img src="https://github.com/Gabvaztor/TFBoost/blob/master/Documentation/Images/Example_Code.png"><br><br>
</div>

"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc."
