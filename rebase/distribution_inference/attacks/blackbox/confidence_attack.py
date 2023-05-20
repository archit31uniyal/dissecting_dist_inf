import torch
import torch.nn

import numpy as np
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions,PredictionsOnOneDistribution,PredictionsOnOneDistribution

class ConfidenceAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False):
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        
        x = preds_adv.preds_on_distr_1
        x_1 = x.preds_property_1
        x_2 = x.preds_property_2
        # x_2 = (num_models, num_points)

        print(x_1)
        print(x_2)

        return [(accs, preds), (None, None), (None,None)]


def confidence_attack(pred_vic1, M_shadow, d_aux):
    """
    Input: M_pretrain, M_t, M_s, d_aux

    Process: Confidence scores M_pretrain -> M_t, M_s0, M_s1 on d_aux
    """
    count_target = 0
    count_shadow = 0
    for img, label in d_aux:
        M_target.eval() 
        M_shadow.eval()
        prob_target = torch.nn.functional.softmax(M_target(img), dim = 1)
        prob_shadow = torch.nn.functional.softmax(M_shadow(img), dim = 1)

        conf_target, class_target = torch.max(prob_target, 1)
        conf_shadow, class_shadow = torch.max(prob_shadow, 1)
        # print(prob_target[0], conf_target)
        # print(class_target[class_target != label])
        # print(class_shadow[class_target != label])
        # print(class_target[class_target == label])

        count_target += class_target[class_target == label].shape[0]
        count_shadow += class_shadow[class_shadow == label].shape[0]

    # count = min(count_target, count_shadow)
    count_diff = abs(count_target - count_shadow)

    count_cent = count_diff/len(d_aux.dataset)

    # return count/len(d_aux.dataset), count_target, count_shadow
    return count_cent, count_target, count_shadow

