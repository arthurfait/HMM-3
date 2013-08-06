# alphabets
EMISSION_ALPHABET G A V P L I M F W Y S T C N Q H D E K R
TRANSITION_ALPHABET  begin F1 F2 B1 B2
MIXTURES Yes 4
########## STATE  begin ###############################################
NAME  begin
LINK  F1 B1
TRANS  uniform
EM_LIST  None
EMISSION  None
MIXTURE_NUM None
MIXTURE_WEIGHTS None
MIXTURE_MEANS None
MIXTURE_COVARS None
ENDSTATE 0
LABEL None
FIX_TR No
FIX_EM No
FIX_EM_MIX No
########## STATE  F1 ###############################################
NAME  F1
LINK  B1 F1
TRANS  uniform
EM_LIST  G A V P L I M F W Y S T C N Q H D E K R
EMISSION  uniform
MIXTURE_NUM 1
MIXTURE_WEIGHTS seq_profile_GMM_data_weights_f
MIXTURE_MEANS seq_profile_GMM_data_means_1_f
MIXTURE_COVARS seq_profile_GMM_data_covars_1_f
ENDSTATE 1
LABEL  f
FIX_TR No
FIX_EM No
FIX_EM_MIX No
########## STATE  F2 ###############################################
NAME  F2
LINK  F2 B2  
TRANS  uniform
EM_LIST  G A V P L I M F W Y S T C N Q H D E K R
EMISSION  uniform
MIXTURE_NUM 1
MIXTURE_WEIGHTS seq_profile_GMM_data_weights_f
MIXTURE_MEANS seq_profile_GMM_data_means_1_f
MIXTURE_COVARS seq_profile_GMM_data_covars_1_f
ENDSTATE 0
LABEL  f
FIX_TR No
FIX_EM No
FIX_EM_MIX No
########## STATE  B1 ###############################################
NAME  B1
LINK  F2 B2
TRANS  uniform
EM_LIST G A V P L I M F W Y S T C N Q H D E K R
EMISSION  uniform
MIXTURE_NUM 1
MIXTURE_WEIGHTS seq_profile_GMM_data_weights_b
MIXTURE_MEANS seq_profile_GMM_data_means_1_b
MIXTURE_COVARS seq_profile_GMM_data_covars_1_b
ENDSTATE 0
LABEL  b
FIX_TR No
FIX_EM No
FIX_EM_MIX No
########## STATE  B2 ###############################################
NAME  B2
LINK  B1 F1
TRANS  uniform
EM_LIST G A V P L I M F W Y S T C N Q H D E K R
EMISSION  uniform
MIXTURE_NUM 1
MIXTURE_WEIGHTS seq_profile_GMM_data_weights_b
MIXTURE_MEANS seq_profile_GMM_data_means_1_b
MIXTURE_COVARS seq_profile_GMM_data_covars_1_b
ENDSTATE 1
LABEL  b
FIX_TR No
FIX_EM No
FIX_EM_MIX No