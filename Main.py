from random import sample
from IR_Component.IR_main import IR_component
from new_DL import DL_component_bleu_threshold
from utils.Evaluator import evaluate_effectiveness_and_save_result_in_xlsx
from utils.tools import load_data, save_data, draw_venn
from decimal import Decimal
from openpyxl import Workbook, load_workbook


def Input(train_body_path, test_body_path, valid_body_path, train_title_path, test_title_path, valid_title_path, train_pred_path, test_pred_path, valid_pred_path):
    train_body = load_data(train_body_path); test_body = load_data(test_body_path); valid_body = load_data(valid_body_path)
    train_title = load_data(train_title_path); test_title = load_data(test_title_path); valid_title = load_data(valid_title_path)
    train_pred = load_data(train_pred_path); test_pred = load_data(test_pred_path); valid_pred = load_data(valid_pred_path)
    return train_body, test_body, valid_body, train_title, test_title, valid_title, train_pred, test_pred, valid_pred


if __name__ == '__main__':
    # give paths
    train_body_path = "iTAPE_dataset/body.train.txt"
    test_body_path = "iTAPE_dataset/body.test.txt"
    valid_body_path = "iTAPE_dataset/body.valid.txt"
    train_title_path = "iTAPE_dataset/title.train.txt"
    test_title_path = "iTAPE_dataset/title.test.txt"
    valid_title_path = "iTAPE_dataset/title.valid.txt"
    train_pred_path = "iTAPE_dataset/body.train.pred.txt"
    test_pred_path = "iTAPE_dataset/body.test.pred.txt"
    valid_pred_path = "iTAPE_dataset/body.valid.pred.txt"

    save_path_DL_reserved_title = "result_temp/DL_reserved_title.txt"
    save_path_DL_reserved_pred = "result_temp/DL_reserved_pred.txt"
    save_path_IR_reserved_title = "result_temp/IR_reserved_title.txt"
    save_path_IR_reserved_pred = "result_temp/IR_reserved_pred.txt"
    save_path_both_reserved_title = "result_temp/both_reserved_title.txt"
    save_path_both_reserved_pred = "result_temp/both_reserved_pred.txt"
    DL_reserved_id_save_path = "result_last/DL_reserved_id.txt"
    IR_reserved_id_save_path = "result_last/IR_reserved_id.txt"

    # input data
    train_body, test_body, valid_body, train_title, test_title, valid_title, train_pred, test_pred, valid_pred = Input(train_body_path, test_body_path, valid_body_path, train_title_path, test_title_path, valid_title_path, train_pred_path, test_pred_path, valid_pred_path)

    # model_names = ['CNN', 'RNN', 'RCNN', 'RNN_Attention', 'Transformer']
    model_names = ['Transformer']
    for model_name in model_names:
        DL_reserved_id = [int(id) for id in DL_component_bleu_threshold('train', model_name, train_body, train_title, train_pred, valid_body, valid_title, valid_pred, test_body, test_title)]
        DL_reserved_pred = [test_pred[int(id)] for id in DL_reserved_id]
        DL_reserved_title = [test_title[int(id)] for id in DL_reserved_id]

        train_plus_validation_body = [body for body in train_body]
        for body in valid_body:
            train_plus_validation_body.append(body)
        IR_reserved_id = [int(id) for id in IR_component(train_body, test_body, test_title, test_pred)]
        IR_reserved_pred = [test_pred[int(id)] for id in IR_reserved_id]
        IR_reserved_title = [test_title[int(id)] for id in IR_reserved_id]

        # save result
        both_reserved_id = list(set(DL_reserved_id) & set(IR_reserved_id))
        both_reserved_title = []
        both_reserved_pred = []
        for idx in both_reserved_id:
            both_reserved_title.append(test_title[int(idx)])
            both_reserved_pred.append(test_pred[int(idx)])
        save_data(DL_reserved_id, DL_reserved_id_save_path)
        save_data(IR_reserved_id, IR_reserved_id_save_path)
        save_data(both_reserved_title, save_path_both_reserved_title)
        save_data(both_reserved_pred, save_path_both_reserved_pred)
        save_data(DL_reserved_pred, save_path_DL_reserved_pred)
        save_data(IR_reserved_pred, save_path_IR_reserved_pred)
        save_data(DL_reserved_title, save_path_DL_reserved_title)
        save_data(IR_reserved_title, save_path_IR_reserved_title)

        # evaluate effectiveness and save result in xlsx
        evaluate_effectiveness_and_save_result_in_xlsx(model_name, test_body_path, test_title_path, test_pred_path,
                                                       DL_reserved_id, IR_reserved_id, save_path_DL_reserved_title,
                                                       save_path_DL_reserved_pred, save_path_IR_reserved_title,
                                                       save_path_IR_reserved_pred, save_path_both_reserved_title,
                                                       save_path_both_reserved_pred)

        # draw venn graph of test set id reserved by DL and IR Component to show intersection of them
        draw_venn(DL_reserved_id, IR_reserved_id)
