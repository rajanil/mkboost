// WARNING: THIS CODE IS ONLY RELEVANT IN THE CONTEXT OF BOOST.PY

double min_loss = 1e10;
int path, feature, threshold;
int D = NX[0];
int N = NX[1];
int P = Nphi_array[0];
int L = NY[0];
int T = Nthresholds[0];
double W_plus, W_minus, W_plus_tilde, W_minus_tilde, Wo;
double loss, W_plus_l, W_minus_l, W_plus_tilde_l, W_minus_tilde_l;

// loop over ADT output nodes
for (int p=0;p<P;p++) {

    // loop over features
    for (int d=0;d<D;d++) {

        double feature_loss = 1e10;
        // loop over thresholds
        for (int t=0;t<T;t++) {

            double thresh = thresholds[t];
            W_plus = 0;
            W_minus = 0;
            W_plus_tilde = 0;
            W_minus_tilde = 0;
            Wo = 0;

            // loop over labels
            for (int l=0;l<L;l++) {

                W_plus_l = 0;
                W_minus_l = 0;
                W_plus_tilde_l = 0;
                W_minus_tilde_l = 0;

                // loop over samples
                for (int n=0;n<N;n++) {
                    if (phi_array[p*N+n]==0)
                        Wo += w[l*N+n];
                    else if (X[d*N+n]>=thresh) {
                        if (Y[l*N+n]==1)
                            W_plus_l += w[l*N+n];
                        else
                            W_minus_l += w[l*N+n];
                    }
                    else {
                        if (Y[l*N+n]==1)
                            W_plus_tilde_l += w[l*N+n];
                        else
                            W_minus_tilde_l += w[l*N+n];
                    }
                }

                if (W_plus_l>=W_minus_l) {
                    W_plus += W_plus_l;
                    W_minus += W_minus_l;
                }
                else {
                    W_plus += W_minus_l;
                    W_minus += W_plus_l;
                }

                if (W_plus_tilde_l>=W_minus_tilde_l) {
                    W_plus_tilde += W_plus_tilde_l;
                    W_minus_tilde += W_minus_tilde_l;
                }
                else {
                    W_plus_tilde += W_minus_tilde_l;
                    W_minus_tilde += W_plus_tilde_l;
                }

            }

            loss = 2*sqrt(W_plus*W_minus) + 
                2*sqrt(W_plus_tilde*W_minus_tilde) + Wo;

            if (loss<min_loss) {
                min_loss = loss;
                threshold = thresh;
                feature = d;
                path = p;
            }
            if (loss<feature_loss) {
                feature_loss = loss;
            }
    
        }
        Z[p*D+d] = feature_loss;
    }
}

results[0] = path;
results[1] = feature;
results[2] = threshold;
