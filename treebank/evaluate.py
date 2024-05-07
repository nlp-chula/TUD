def evaluate_conllu(*, pred_path: str, gold_path: str):
    """
    return UAS and LAS respectively
    """
    arg_correct = 0
    arg_label_correct = 0
    total = 0
    with open(pred_path, encoding="utf-8") as pred_file,\
        open(gold_path, encoding="utf-8") as gold_file:
        for pred, gold in zip(pred_file, gold_file):
            if pred == '\n' or pred.startswith('#'):
                assert pred == gold, f"File mismatch:\n{pred}\n{gold}"
                continue
            _, _, _, _, _, _, arg_pred, label_pred, _, _ = pred.split('\t')
            _, _, _, _, _, _, arg_gold, label_gold, _, _ = gold.split('\t')
            total += 1
            if arg_pred == arg_gold:
                arg_correct += 1
                if label_pred == label_gold:
                    arg_label_correct += 1
    uas = arg_correct / total
    las = arg_label_correct / total
    return uas, las
