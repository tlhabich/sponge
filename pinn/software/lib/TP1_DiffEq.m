function u_filt = TP1_DiffEq(u_raw, u_prev, a, y_prev)
u_filt = ( -(1-a)*y_prev + u_raw + u_prev) / (1+a);
