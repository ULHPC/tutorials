#ifndef _FHN_HPP_
#define _FHN_HPP_
class FHN {
public:
    const static int SYS_SIZE = 2;
    const static int COUPLING_VAR_ID = 0;
    double Y[SYS_SIZE];
    double a;
    double epsilon;

    // double stim_amplitude;   // microA_per_cm2 (in membrane)
    // double stim_duration;   // millisecond (in membrane)
    // double stim_end;   // millisecond (in membrane)
    // double stim_period;   // millisecond (in membrane)
    // double stim_start;   // millisecond (in membrane)

    void init() {}
    void compute(double time, double *out) {
        // double I_stim = 0;
        // if ((time >= stim_start) && (time <= stim_end) && (time-stim_start-floor((time-stim_start)/stim_period)*stim_period <= stim_duration))
            // I_stim = stim_amplitude;
        // else
            // I_stim = 0.0;

        out[0] = Y[0] - Y[0]*Y[0]*Y[0]/3.0 - Y[1];
        out[1] = epsilon*(Y[0] + a);
    }

    double get_var(int i) {
        return Y[i];
    }
    FHN() {
        epsilon = 0.02;
        a = 1.2;
        Y[0] = -a;
        Y[1] = Y[0] - Y[0]*Y[0]*Y[0]/3.0;
    }
};
#endif
