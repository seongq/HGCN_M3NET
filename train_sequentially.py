import subprocess

# 설정할 파라미터 조합
curvatures = [1,-10,-0.1,-0.01,-5.0, 1,-1,-7,  -20,0.1, 0.5, -5,-1,-0.1,-0.01,-0.001,-0.0001,-0.00001,0,0.00001,0.0001,0.001,0.01,0.1,1,5,10]
num_Ks = [1,2,3,4,5,6,7]
num_Ls = [1,2,3,4,5,6,7]
graph_types = ['hyperbolicGCNSingle', "hyperbolicGCNSingle_hypergraph","hyperbolicGCNSingle_highfreq",   "hyperbolicGCN_highfreq_hypergraph"]
datasets = ['MELD']

# 기본 명령어 설정
base_command = ("CUDA_VISIBLE_DEVICES=1 python -u train.py --base-model 'GRU' --lr 0.0001 "
                "--batch-size 16 --epochs=30 --graph_construct='direct' --multi_modal "
                "--mm_fusion_mthd='concat_DHT' --modals='avl' --norm BN ")

# 파라미터 조합을 순회하며 명령어 실행
for graph_type in graph_types:
    for num_L in num_Ls:
        for num_K in num_Ks:
            for curvature in curvatures:
                
                    for dataset in datasets:
                        for i in range(10):
                            dropout = 0.5 if dataset == 'IEMOCAP' else 0.4
                            output_file = f"test_{graph_type}_K_{num_K}_curvature_{curvature}_{dataset}.txt"
                            command = (f"{base_command} --dropout {dropout} --graph_type {graph_type} --num_K {num_K} --num_L {num_L} --curvature {curvature} "
                                    f"--Dataset {dataset} ")
                            print(f"Running command: {command}")
                            subprocess.run(command, shell=True)
