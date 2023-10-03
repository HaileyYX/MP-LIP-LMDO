# MP-LIP-LMDO

-问题：考虑配送外包的选址-库存问题

Problem: location and inventory problem considering last-mile delivery outsourcing

-算法：自适应大规模邻域搜索算法ALNS

Algorithm: Adaptive Large Neighborhood Search Algorithm (ALNS)

## 备注

## Note

-在main中运行文件

Running this procedure in the file named main.

-data_loading为算例数据导入，需要时修改以下代码中的“Instance10"文件名即可，数据文件为xlsx格式，10代表备件数量。导入数据需要同步修改data1中备件数量，否则会报错

The file named data_loading is used for importing instance data. If needed, you should modify the filename "Instance10" in the code below, the data file is in xlsx format, and the number of  10 represents the number of types of spare parts. The number of spare parts in 'data1' needs to be modified synchronously while importing data; otherwise, an error will occur.

def data_loading(pr: Parameter):

df_data = pd.read_excel("Instance10.xlsx", sheet_name='data')

df_data_t = pd.read_excel('Instance10.xlsx', sheet_name='data_t_jk')

df_data_h = pd.read_excel('Instance10.xlsx', sheet_name='data_h_jk')

df_data_info_cust = pd.read_excel('Instance10.xlsx', sheet_name='info_cust')

-Instance10为小规模算例测试集示例，便于理解如何制作需要的实例数据集

The file named Instance10 is a small-scale instance test set example, which facilitates understanding how to create the required instance data set.

-parameters中包含算法使用的参数，需要时对应修改

The file named Parameters contains the parameters used by the proposed ALNS algorithm, and you should modify these parameters when you have a necessary.

-classes、fuction_tools为代码使用定义的类、算子等，无须修改；配置在python同一个文件夹即可

The files named Classes and function_tools respectively are classes, operators whicn are defined for our procedure, you have no necessary to modify them. You just need to configure them in the same folder as the Python file.
