using Microsoft.ML.Data;

namespace ML_Chapter_2.ML.Objects
{
    public class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float DurationInMonths;
    }
}