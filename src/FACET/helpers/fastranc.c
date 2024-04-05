#include <stdlib.h>
#include <math.h>

void fastranc(double *refs, double *d, int N, double mu, double *out, double *y, int veclength) {
    double *W;
    int i, j;
    W = (double *)malloc((N+1) * sizeof(double));
    double temp;
    
    for (i = 0; i <= N; i++) {
        W[i] = 0;
    }
    
    for (i = 0; i < veclength; i++) {
        if (i < N) { 
            out[i] = 0; y[i] = 0;
        } else {
            y[i] = 0;
            
            for (j = 0; j <= N; j++) {
                y[i] += W[j] * refs[i - N + j]; 
            }
            
            out[i] = d[i] - y[i];
            
            temp = 2 * mu * out[i];
            for (j = 0; j <= N; j++) {
                W[j] += temp * refs[i - N + j];
            }
        }
    }
    
    free(W);
}