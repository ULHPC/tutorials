#ifndef __LR_I_HPP__
#define __LR_I_HPP__
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

//==============================================================================
// CellML file:   luo_rudy_1991.cellml
// CellML model:  luo_rudy_1991
// Date and time: 20.01.2015 at 9:12:08
//------------------------------------------------------------------------------
// Conversion from CellML 1.0 to C++ (header) was done using COR (0.9.31.1409)
//    Copyright 2002-2015 Dr Alan Garny
//    http://cor.physiol.ox.ac.uk/ - cor@physiol.ox.ac.uk
//------------------------------------------------------------------------------
// http://www.cellml.org/
//==============================================================================


class LR_I {
public:
    const static int SYS_SIZE = 8;
    const static int COUPLING_VAR_ID = 4;

    //------------------------------------------------------------------------
    // State variables
    //------------------------------------------------------------------------
    double Y[SYS_SIZE];
    // double dY[SYS_SIZE];
    // 0: h (dimensionless) (in fast_sodium_current_h_gate)
    // 1: j (dimensionless) (in fast_sodium_current_j_gate)
    // 2: m (dimensionless) (in fast_sodium_current_m_gate)
    // 3: Cai (millimolar) (in intracellular_calcium_concentration)
    // 4: V (millivolt) (in membrane)
    // 5: d (dimensionless) (in slow_inward_current_d_gate)
    // 6: f (dimensionless) (in slow_inward_current_f_gate)
    // 7: X (dimensionless) (in time_dependent_potassium_current_X_gate)


    double E_b;   // millivolt (in background_current)
    double g_b;   // milliS_per_cm2 (in background_current)
    double g_Na;   // milliS_per_cm2 (in fast_sodium_current)
    double Ki;   // millimolar (in ionic_concentrations)
    double Ko;   // millimolar (in ionic_concentrations)
    double Nai;   // millimolar (in ionic_concentrations)
    double Nao;   // millimolar (in ionic_concentrations)
    double C;   // microF_per_cm2 (in membrane)
    double F;   // coulomb_per_mole (in membrane)
    double R;   // joule_per_kilomole_kelvin (in membrane)
    double T;   // kelvin (in membrane)
    double stim_amplitude;   // microA_per_cm2 (in membrane)
    double stim_duration;   // millisecond (in membrane)
    double stim_end;   // millisecond (in membrane)
    double stim_period;   // millisecond (in membrane)
    double stim_start;   // millisecond (in membrane)
    double g_Kp;   // milliS_per_cm2 (in plateau_potassium_current)
    double PR_NaK;   // dimensionless (in time_dependent_potassium_current)


    double i_b;   // microA_per_cm2 (in background_current)
    double alpha_h;   // per_millisecond (in fast_sodium_current_h_gate)
    double beta_h;   // per_millisecond (in fast_sodium_current_h_gate)
    double alpha_j;   // per_millisecond (in fast_sodium_current_j_gate)
    double beta_j;   // per_millisecond (in fast_sodium_current_j_gate)
    double alpha_m;   // per_millisecond (in fast_sodium_current_m_gate)
    double beta_m;   // per_millisecond (in fast_sodium_current_m_gate)
    double E_Na;   // millivolt (in fast_sodium_current)
    double i_Na;   // microA_per_cm2 (in fast_sodium_current)
    double I_stim;   // microA_per_cm2 (in membrane)
    double E_Kp;   // millivolt (in plateau_potassium_current)
    double Kp;   // dimensionless (in plateau_potassium_current)
    double i_Kp;   // microA_per_cm2 (in plateau_potassium_current)
    double alpha_d;   // per_millisecond (in slow_inward_current_d_gate)
    double beta_d;   // per_millisecond (in slow_inward_current_d_gate)
    double alpha_f;   // per_millisecond (in slow_inward_current_f_gate)
    double beta_f;   // per_millisecond (in slow_inward_current_f_gate)
    double E_si;   // millivolt (in slow_inward_current)
    double i_si;   // microA_per_cm2 (in slow_inward_current)
    double alpha_X;   // per_millisecond (in time_dependent_potassium_current_X_gate)
    double beta_X;   // per_millisecond (in time_dependent_potassium_current_X_gate)
    double Xi;   // dimensionless (in time_dependent_potassium_current_Xi_gate)
    double E_K;   // millivolt (in time_dependent_potassium_current)
    double g_K;   // milliS_per_cm2 (in time_dependent_potassium_current)
    double i_K;   // microA_per_cm2 (in time_dependent_potassium_current)
    double K1_infinity;   // dimensionless (in time_independent_potassium_current_K1_gate)
    double alpha_K1;   // per_millisecond (in time_independent_potassium_current_K1_gate)
    double beta_K1;   // per_millisecond (in time_independent_potassium_current_K1_gate)
    double E_K1;   // millivolt (in time_independent_potassium_current)
    double g_K1;   // milliS_per_cm2 (in time_independent_potassium_current)
    double i_K1;   // microA_per_cm2 (in time_independent_potassium_current)
    double g_Si;

    void init();
    void compute(double time, double *out);

    double get_var(int i) {
        return Y[i];
    }

    LR_I() {
        E_b = -59.87;   // millivolt (in background_current)
        g_b = 0.03921;   // milliS_per_cm2 (in background_current)
        g_Na = 23.0;   // milliS_per_cm2 (in fast_sodium_current)
        Ki = 145.0;   // millimolar (in ionic_concentrations)
        Ko = 5.4;   // millimolar (in ionic_concentrations)
        Nai = 18.0;   // millimolar (in ionic_concentrations)
        Nao = 140.0;   // millimolar (in ionic_concentrations)
        C = 1.0;   // microF_per_cm2 (in membrane)
        F = 96484.6;   // coulomb_per_mole (in membrane)
        R = 8314.0;   // joule_per_kilomole_kelvin (in membrane)
        T = 310.0;   // kelvin (in membrane)
        stim_amplitude = 0; //-25.5;   // microA_per_cm2 (in membrane)
        stim_duration = 2.0;   // millisecond (in membrane)
        stim_end = 9000.0;   // millisecond (in membrane)
        stim_period = 1000.0;   // millisecond (in membrane)
        stim_start = 100.0;   // millisecond (in membrane)
        g_Kp = 0.0183;   // milliS_per_cm2 (in plateau_potassium_current)
        PR_NaK = 0.01833;   // dimensionless (in time_dependent_potassium_current)


        E_Na = R*T/F*log(Nao/Nai);
        g_K = 0.9*sqrt(Ko/5.4);
        E_K = R*T/F*log((Ko+PR_NaK*Nao)/(Ki+PR_NaK*Nai));
        g_K1 = 0.6047*sqrt(Ko/5.4);
        E_K1 = R*T/F*log(Ko/Ki);
        E_Kp = E_K1;
        g_Si = 0.03;
    }
};
#endif

