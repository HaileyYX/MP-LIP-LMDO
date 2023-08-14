# MP-LIP-LMDO

-问题：考虑配送外包的选址-库存问题

-算法：自适应大规模邻域搜索算法ALNS

## 备注

-在main中运行文件

-data_loading为算例数据导入，需要时修改以下代码中的“Instance10"文件名即可，数据文件为xlsx格式，10代表备件数量。导入数据需要同步修改data1中备件数量，否则会报错

def data_loading(pr: Parameter):
df_data = pd.read_excel("Instance10.xlsx", sheet_name='data')
df_data_t = pd.read_excel('Instance10.xlsx', sheet_name='data_t_jk')
df_data_h = pd.read_excel('Instance10.xlsx', sheet_name='data_h_jk')
df_data_info_cust = pd.read_excel('Instance10.xlsx', sheet_name='info_cust')

-Instance10为小规模算例测试集示例，便于理解如何制作需要的实例数据集

-parameters中包含算法使用的参数，需要时对应修改

-classes、fuction_tools为代码使用定义的类、算子等，无须修改；配置在python同一个文件夹即可
