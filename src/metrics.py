def calculate_ca_corr(clean_accuracy):
    """
    Chance-Corrected Accuracy.
    AG News has 4 balanced classes, so random chance is 0.25.
    """
    if clean_accuracy <= 0.25:
        return 0.0
    return max(0.0, (clean_accuracy - 0.25) / 0.75)

def calculate_asr(predictions, target_label=1):
    """
    Attack Success Rate on the triggered test set.
    """
    if not predictions:
        return 0.0
        
    target_count = sum(1 for p in predictions if p == target_label)
    return target_count / len(predictions)


def calculate_ftr(clean_predictions, clean_true_labels, target_label=1):
    """
    False Trigger Rate.
    Calculates the percentage of non-target clean samples that incorrectly 
    predicted the target label.
    """
    if not clean_predictions:
        return 0.0
        
    # Find all indices where the true label is NOT the target label
    non_target_indices = [i for i, true_label in enumerate(clean_true_labels) if true_label != target_label]
    
    if not non_target_indices:
        return 0.0
        
    # Count how many of those non-target samples actually predicted the target label
    false_triggers = sum(1 for i in non_target_indices if clean_predictions[i] == target_label)
    
    return false_triggers / len(non_target_indices)


def check_collapse(ca_corr, ftr):
    """
    Dead-Model Collapse Guard.
    Returns True if the model is broken (FTR >= 50% or CA_corr <= 0).
    """
    if ftr >= 0.50 or ca_corr <= 0.0:
        return True
    return False    

def calculate_differential_persistence(asr_quant, asr_f16, ca_quant, ca_f16, is_collapsed):
    """
    Differential Persistence.
    Calculates the difference in retention ratios between ASR and Clean Accuracy.
    """
    if is_collapsed:
        return float('nan')
        
    # Safety check: if the F16 baseline failed completely, we cannot divide by zero
    if asr_f16 == 0 or ca_f16 == 0:
        return float('nan')
        
    r_asr = asr_quant / asr_f16
    r_ca = ca_quant / ca_f16
    
    differential = r_asr - r_ca
    return differential