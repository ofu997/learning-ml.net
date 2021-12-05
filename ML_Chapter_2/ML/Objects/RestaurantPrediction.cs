﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Chapter_2.ML.Objects
{
    public class RestaurantPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
