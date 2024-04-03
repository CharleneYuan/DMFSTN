# DMFSTN

## 使用说明
主要文件夹为 *libcity* 文件夹，模型位于*libcity/model/traffic_speed_prediction/* 下，DMFSTN.py 为含有分解模式的模型，MFSTN.py为不含有分解模式的模型。
配置文件位于 *libcity/config/model/traffic_state_pred/* 下，可更改参数。
依赖库与libcity一致。

## run model
run_model.py文件，可更改 *model* 后参数调整模型；可更改 *dataset* 调整数据集；可更改 *exp_id* 用于指定运行id。

## 训练中止
如训练中止需评估模型，可运行 *evaluate.py* 指定 *exp_id* 和最佳 *epoch* 评估当前最优模型。

