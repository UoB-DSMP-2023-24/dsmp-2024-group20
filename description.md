# ARIMA
## basic concepts
**AR(AutoRegressive)**:  uses past values of the series to predict future values. The notation AR(p) indicates that the model uses p past observations to predict the current one.  

**I(Integrated)**:   The integrated component in ARIMA refers to the differencing step that is applied to make the time series stationary.  

**MA(Moving Average)**:  The moving average component of ARIMA model uses the errors from the autoregressive models to predict the current value. MA(q) indicates that the model uses q past errors to predict the current one. parameter(d)

**ACF(auto correlation function)**: reflects the correlation between observations of a time series at different time points. The ACF can range in value from -1 to 1 (from negative correlation to positive correlation).  

**PACF(partial auto correlation function)**: only reflects the correlation of x(t) and x(t-k), but ACF also contains the influence of x(t-1), x(t-2), ... , x(t-k+1)

## How to find p, q, d?
p: when does PACF fall into the 95% confidence interval.  
q: when does ACF fall into the 95% confidence interval.     
d: direct observation of several orders of stability.   
**or**  
we can iterate a range of variables and see which one gives us the best result.
