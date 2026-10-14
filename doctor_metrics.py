import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix

# ========== 在这里填写你的Excel路径 ==========
input_excel = "/home/uax/SCY/HSQ/HSQformer/doctor_per_image_data.xlsx"

# 读取excel
df = pd.read_excel(input_excel)

# 金标准标签 y_true：列名完全原样复制
y_true = df["last_diagnosis_image(0=benign;1=malignant)"].values

# 待评估的列：6位医生【无AI】 + 6位医生【with AI】，列名严格匹配
eval_cols = [
    ("senior_1", "senior_1_dignosis_image"),
    ("senior_2", "senior_2_dignosis_image"),
    ("senior_2_withAI", "senior_2_dignosis_image_with_AI"),
    ("senior_3", "senior_3_dignosis_image"),
    ("senior_3_withAI", "senior_3_dignosis_image_with_AI"),
    ("junior_1", "junior_1_dignosis_image"),
    ("junior_1_withAI", "junior_1_dignosis_image_with_AI"),
    ("junior_2", "junior_2_dignosis_image"),
    ("junior_2_withAI", "junior_2_dignosis_image_with_AI"),
    ("junior_3", "junior_3_dignosis_image"),
    ("junior_3_withAI", "junior_3_dignosis_image_with_AI"),
    ("AI_image", "AI_image")
]

# 存储结果
result_list = []

for name, col in eval_cols:
    y_pred = df[col].values
    
    # 混淆矩阵 tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred) # 敏感性=Recall
    specificity = tn / (tn + fp) if (tn+fp) !=0 else np.nan
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        auc = np.nan

    res = {
        "Reader": name,
        "Accuracy": acc * 100,
        "Precision": precision * 100,
        "Recall": sensitivity * 100,
        "Specificity": specificity * 100,
        "F1": f1 * 100,
        "AUC": auc * 100 if not np.isnan(auc) else np.nan,
        "TP":tp, "TN":tn, "FP":fp, "FN":fn
    }
    result_list.append(res)

# 额外单独计算 LiverNet_M_probability 概率的AUC
y_prob = df["LiverNet_M_probability"].values
try:
    auc_livernet = roc_auc_score(y_true, y_prob)
    res_livernet = {
        "Reader": "LiverNet_M_prob",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "Specificity": np.nan,
        "F1": np.nan,
        "AUC": auc_livernet * 100,
        "TP":np.nan, "TN":np.nan, "FP":np.nan, "FN":np.nan
    }
    result_list.append(res_livernet)
except:
    pass

metrics_df = pd.DataFrame(result_list)

# ===================== 分组均值±标准差（无AI组）=====================
# 筛选无AI的senior1/2/3 和 junior1/2/3
senior_names = ["senior_1","senior_2","senior_3"]
junior_names = ["junior_1","junior_2","junior_3"]
all_doctor_names = senior_names + junior_names  #全部6位医生（无AI）

senior_df = metrics_df[metrics_df["Reader"].isin(senior_names)]
junior_df = metrics_df[metrics_df["Reader"].isin(junior_names)]
all_doctor_df = metrics_df[metrics_df["Reader"].isin(all_doctor_names)]

metrics_cols = ["Accuracy","Precision","Recall","Specificity","F1","AUC"]

def calc_mean_std(sub_df):
    mean_dict = {}
    for m in metrics_cols:
        vals = sub_df[m].dropna()
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1) #样本标准差 ddof=1（论文常用）
        mean_dict[m] = f"{mean_val:.2f} ± {std_val:.2f}"
    return mean_dict

senior_avg = calc_mean_std(senior_df)
senior_avg["Reader"] = "senior_average"

junior_avg = calc_mean_std(junior_df)
junior_avg["Reader"] = "junior_average"

all_doctor_avg = calc_mean_std(all_doctor_df)
all_doctor_avg["Reader"] = "all_doctors_average"

# 将均值行追加到结果表
metrics_df = pd.concat([
    metrics_df,
    pd.DataFrame([senior_avg]),
    pd.DataFrame([junior_avg]),
    pd.DataFrame([all_doctor_avg])
], ignore_index=True)


# ========== 输出 ==========
print("======= 诊断性能指标结果（百分比形式） =======")
print(metrics_df)

# 保存csv
metrics_df.to_csv("doctor_metrics_result_with_all_avg.csv", index=False, encoding="utf-8-sig")
print("\n结果（含senior/junior/all_doctors均值±标准差）已保存到 doctor_metrics_result_with_all_avg.csv")
