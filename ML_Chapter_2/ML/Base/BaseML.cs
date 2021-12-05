using Microsoft.ML;
using ML_Chapter_2.Common;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ML_Chapter_2.ML.Base
{
    //The BaseML class, as discussed earlier, is going to contain the common code between our
    //Trainer and Predictor classes, starting with this chapter
    public class BaseML
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);
        //For all ML.NET applications in both training and predictions, an MLContext object is
        //required
        protected readonly MLContext MlContext;

        protected BaseML()
        {
            MlContext = new MLContext(2020);
        }
    }
}
