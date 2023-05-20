from distribution_inference.attacks.blackbox.analyze_outputs import *

def JI_attack(M_target, M_shadow_M0, M_shadow_M1):
    """
    JI_threshold = 0.3679
    """
    ji_M0, _, _ = calculate_JI(M_target, M_shadow_M0, 'conv', 1)
    ji_M1, _, _ = calculate_JI(M_target, M_shadow_M1, 'conv', 1)

    ji_target_M0 = get_JI(ji_M0)
    ji_target_M1 = get_JI(ji_M1)

    if ji_target_M0 > ji_target_M1:
        result = 0
    else:
        result = 1
    # JI_exp = np.exp(-(ji_target_M0/ji_target_M1))
    # print(ji_target_M0, ji_target_M1)
    return result, ji_target_M0, ji_target_M1

