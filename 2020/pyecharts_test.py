# encoding: utf-8
'''
# @Author  : ccq
# @File    : demo11.py
# @Time    : 2019/7/29 17:20
'''

import pyecharts.options as opts
from pyecharts.charts import Line


#from example.commons import Faker

from pyecharts.charts import Map  # 导入模块

from pyecharts.charts import Geo  # 导入模块

# 设置数据
persondata = [('云南', '1'), ('云南', '3'), ('北京', '4'), ('甘肃', '4'), ('广西', '4'), ('广西', '3'), ('广西', '6'),
              ('广东', '4'), ('广东', '1'), ('广东', '2'), ('安徽', '4'), ('贵州', '4'), ('河南', '4'), ('湖北', '5'),
              ('山西', '4'), ('陕西', '4'), ('上海', '4'), ('四川', '4'), ('四川', '5'), ('四川', '3'), ('四川', '6'),
              ('新疆', '3'), ('新疆', '4'), ('浙江', '4'), ('浙江', '5'), ('浙江', '6'), ('云南', '5'), ('云南', '3'),
              ('重庆', '4'), ]
chickendata = [('北京', '4'), ('广西', '4'), ('广西', '3'), ('河南', '4'), ('河南', '3'), ('江苏', '4'), ('江西', '4'),
               ('江西', '5'), ('山东', '4'), ('上海', '4'), ('四川', '4'), ('四川', '5'), ('四川', '3'), ('广东', '4'), ('广东', '3'), ]
duckdata = [('北京', '4'), ('山东', '4'), ('浙江', '4'), ('广东', '4'), ('广东', '3'), ]
pigdata = [('广西', '4'), ('广西', '5'), ('广东', '4'), ('黑龙江', '4'), ('江西', '4'), ('江西', '5'), ('山东', '4'), ]
seafooddata = [('广东', '4'), ]
mousedata = [('河南', '4'), ]
cattledata = [('新疆', '3'), ]
sheepdata = [('新疆', '3'), ]
geo = Geo()  # 初始化配置项
geo.add_schema(maptype="china")  # 设置地图类型
geo.add(  # 添加图例
    '鼠',  # 图例名称
    persondata,  # 数据源
    symbol='circle',  # 图例形状

)
geo.add(
    '牛',
    chickendata,
    symbol='rect',
)
geo.add(
    '虎',
    duckdata,
    symbol='roundRect',
)
geo.add(
    '兔',
    pigdata,
    symbol='triangle',
)
geo.add(
    '龙',
    seafooddata,
    symbol='diamond',
)
geo.add(
    '蛇',
    mousedata,
    symbol='pin',
)
geo.add(
    '马',
    cattledata,
    symbol='arrow',
)
geo.add(
    '羊',
    sheepdata,
    symbol='none',
)
geo.set_global_opts(  # 设置全局项

    title_opts=opts.TitleOpts(  # 设置标题配置项
        title="中国地图",  # 设置标题名称
        pos_left="center"  # 设置标题距离容器左边的位置 这里为居中
    ),
    visualmap_opts=opts.VisualMapOpts(  # 设置视觉映射配置项
        is_piecewise=True,  # 设置是否为分段型

        pos_left="left",  # 设置视觉映射距离容器左边的位置 这里为居左
        pos_bottom="bottom",  # 设置视觉映射距离容器底部的位置 这里为底部
        orient="vertical",  # 设置水平（'horizontal'）或者竖直（'vertical'）
        pieces=[  # 设置每段的范围、文字、颜色
            {"value": "1", "label": "A", "color": "red"},
            {"value": "2", "label": "B", "color": "orange"},
            {"value": "3", "label": "C", "color": "yellow"},
            {"value": "4", "label": "D", "color": "green"},
            {"value": "5", "label": "E", "color": "blue"},
            {"value": "6", "label": "F", "color": "cyan"},
            {"value": "7", "label": "G", "color": "purple"}
        ],

    ),
    legend_opts=opts.LegendOpts(  # 设置图例配置项
        pos_right="right",
        pos_top="top",
        orient="vertical",

    ),
)
geo.set_series_opts(
    label_opts=opts.LabelOpts(  # 设置标签配置项
        is_show=False  # 设置不显示Label
    )
)

geo.render("demo03.html")  # 生成名为"demo03"的本地html文件

# customMap = (
#     Map()
#         .add("商家A",  # 图例
#              [list(z) for z in zip(Faker.provinces, Faker.values())],  # 数据项
#              "china"  # 地图
#              )
#         .set_global_opts(  # 设置全局项
#         title_opts=opts.TitleOpts(  # 设置标题项
#             title="中国地图"  # 设置标题名称
#         )
#     )
# )
# customMap.render("demo11.html")  # 生成名为demo11的本地html文件


line=Line()
line.add_xaxis(["201{}年/{}季度".format(y,z)
for y in range(4)
for z in range(1,5)]) #设置x轴数据
line.add_yaxis(
"电视机销量",
[4.80,4.10,6.00,6.50,5.80,5.20,6.80,7.40,
6.00,5.60,7.50,7.80,6.30,5.90,8.00,8.40]
)#设置y轴数据
line.set_global_opts(
xaxis_opts=opts.AxisOpts(
axislabel_opts=opts.LabelOpts(rotate=-40),
),#设置x轴标签旋转角度
yaxis_opts=opts.AxisOpts(name="销量（单位/千台）"),#设置y轴名称
title_opts=opts.TitleOpts(title="折线图")) #设置图表标题

line.render_notebook() #渲染图表

