
import openke
from openke.config import myTester
from openke.module.model import AttMBertTransR
from openke.data import BertTrainDataLoader,BertTestDataLoader
import os
import statistics
import json



IN_PATH= "/home/tianyi/codeProjects/con_exploit/mydata/Openke_test/wo_abstraction_label_inverse_kgc/"
OUT_PATH ="/home/tianyi/Desktop/kgprj/kg-reason code/paper_code/get_reason_data/cweChainData/"
if not os.path.exists(IN_PATH + 'checkpoint/'):
    os.makedirs(IN_PATH + 'checkpoint/')




def att_mberttransr():
    max_num_tokens = 50
    train_dataloader = BertTrainDataLoader(
        in_path=IN_PATH,
        nbatches=80,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=4,
        neg_rel=0,max_num_tokens=max_num_tokens)

    # dataloader for test
    test_dataloader = BertTestDataLoader(
        in_path=IN_PATH,
        sampling_mode='link',type_constrain = False,max_num_tokens=max_num_tokens)

    # define the model
    transr = AttMBertTransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=100,
        dim_r=100,
        p_norm=1,
        norm_flag=True,
        rand_init=False,max_num_tokens=max_num_tokens)


    # test the model
    transr.load_checkpoint(IN_PATH+'checkpoint/AttMBertTransR.ckpt')
    tester = myTester(model=transr, data_loader=test_dataloader, use_gpu=True)
    kgcdict=tester.run_link_prediction(type_constrain=False,kgcFlag=True)
    with open(IN_PATH + "att_mberttransr_kgc_index.json", "w") as f:
        json.dump(kgcdict, f,indent=3)



def kgc_data_deduplication(dedupFlag=True):
    with open(IN_PATH + "att_mberttransr_kgc_index.json", "r") as f:
        kgcdict = json.load(f)
    with open(OUT_PATH+"cwe_relate_cve.json","r") as f:
        cwe_cve_dict=json.load(f)
    with open(OUT_PATH+"18-22_cwe_relate_cwe_CanPrecede_StartsWith.json","r") as f:
        cwe_cwe_dict=json.load(f)
    with open(IN_PATH + "entity2id.txt", "r") as f:
        temList = f.readlines()
        temList.remove(temList[0])
        f.close()
    id2entity = [None for i in range(len(temList))]
    for item in temList:
        item = item.strip().split()
        id2entity[int(item[1])] = item[0]
    with open(IN_PATH+"relation2id.txt","r") as f:
        temList = f.readlines()
        temList.remove(temList[0])
        f.close()
    id2relation = [None for i in range(len(temList))]
    for item in temList:
        item = item.strip().split()
        id2relation[int(item[1])] = item[0]
    kgcdict_after= {}
    for headid in kgcdict:
        for relid in kgcdict[headid]:
            for tailTuple in kgcdict[headid][relid]:
                head = id2entity[int(headid)]
                tail = id2entity[int(tailTuple[0])]
                rel = id2relation[int(relid)]
                if head == tail:
                    continue
                if(dedupFlag):
                    if("CVE" in tail):
                        if head in cwe_cve_dict and tail in cwe_cve_dict[head]:
                            continue
                    else:
                        if head in cwe_cve_dict and rel in cwe_cve_dict[head] and tail in cwe_cve_dict[head][rel]:
                            continue
                if head not in kgcdict_after:
                    kgcdict_after[head] = {}
                if rel not in kgcdict_after[head]:
                    kgcdict_after[head][rel] = []
                kgcdict_after[head][rel].append((tail,tailTuple[1]))
    if(dedupFlag):
        with open(IN_PATH + "att_mberttransr_kgc_deduplication.json", "w") as f:
            json.dump(kgcdict_after, f,indent=3)
    else:
        with open(IN_PATH + "att_mberttransr_kgc.json", "w") as f:
            json.dump(kgcdict_after, f, indent=3)


def get_statistics_and_top_percent_values(data, percent):
    """
    获取列表的统计数据以及前百分比的数值阈值。

    参数:
    data (list): 输入的数值列表。
    percent (float): 需要提取的百分比（0 到 1 之间）。

    返回:
    dict: 包含统计数据和前百分比数值的字典。
    """
    if not data:
        raise ValueError("data 不能为空列表。")
    if not 0 < percent <= 1:
        raise ValueError("percent 必须在 0 到 1 之间。")

    # 对列表进行排序
    sorted_data = sorted(data)

    # 计算百分比的索引
    n = len(data)
    index = int(n * percent)-1

    # 获取前百分比的数值
    top_percent_value = sorted_data[index]

    # 计算统计数据
    mean = statistics.mean(data)
    median = statistics.median(data)
    variance = statistics.variance(data)
    stdev = statistics.stdev(data)
    maximum = max(data)
    minimum = min(data)
    total_sum = sum(data)
    count = len(data)

    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "stdev": stdev,
        "max": maximum,
        "min": minimum,
        "sum": total_sum,
        "count": count,
        "top_percent_value": top_percent_value
    }
#得到最终的kgc数据
def genkgc_data():
    with open(IN_PATH + "att_mberttransr_kgc_deduplication.json", "r") as f:
        dedup_kgcdict = json.load(f)
    with open(IN_PATH + "att_mberttransr_kgc.json", "r") as f:
        kgcdict = json.load(f)
    disScoresList={}
    for head in kgcdict:
        for rel in kgcdict[head]:
            for tailTuple in kgcdict[head][rel]:
                if rel not in disScoresList:
                    disScoresList[rel]=[]
                disScoresList[rel].append(tailTuple[1])
    threshold =5.8
    for rel in disScoresList:
        print(rel+"    ")
        res=get_statistics_and_top_percent_values(disScoresList[rel],0.3)
        threshold= min(res["top_percent_value"],threshold)
        print(res)
    kgcdict_after = {} # 存储最终的kgc数据
    for head in dedup_kgcdict:
        for rel in dedup_kgcdict[head]:
            if rel not in kgcdict_after:
                kgcdict_after[rel] = []
            for tailTuple in dedup_kgcdict[head][rel]:
                if tailTuple[1] < threshold:
                    if(rel=='isInstanceOf_inverse' and "CWE" in tailTuple[0]):
                        continue
                    kgcdict_after[rel].append((head,tailTuple[0]))
    with open(OUT_PATH + "att_mberttransr_kgc_final.json", "w") as f:
        json.dump(kgcdict_after, f, indent=3)

if __name__ == "__main__":
    # att_mberttransr()
    # kgc_data_deduplication(dedupFlag=False)
    genkgc_data()
    print("#####################All Test Done##########################")










