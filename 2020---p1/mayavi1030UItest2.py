from numpy import cos,sin,pi,arange
from traits.api import HasTraits,Instance,Range,on_trait_change
from traitsui.api import View,Item,Group
from mayavi.core.ui.api import MayaviScene,SceneEditor,MlabSceneModel
from mayavi.core.api import PipelineBase

dphin = pi/300.
phi = arange(0.0,2*pi+0.5*dphin,dphin,'d')
#建立数据
def curve(n_mer,n_long):
    mu = phi*n_mer
    x = cos(mu)*(1+cos(n_long/n_mer)*0.5)
    y = sin(mu)*(1+cos(n_long/n_mer)*0.5)
    z = 0.5*sin(n_long*mu/n_mer)
    t = sin(mu)
    return x,y,z,t

class MyModel(HasTraits):
    n_meridional = Range(0,30,6)
    n_longitudinal = Range(0,30,11)
#场景模型实例
    scene = Instance(MlabSceneModel,()) #后面加上()是将他实例化了
#管线实例
    plot = Instance(PipelineBase)

    def __init__(self,**traits):
        HasTraits.__init__(self,**traits)
        x, y, z, t = curve(self.n_meridional, self.n_longitudinal)
        if self.plot is None: # 如果plot未绘制则输出plot3d
            self.plot = self.scene.mlab.plot3d(x, y, z, t,
                                               tube_radius=0.025, colormap="Spectral")

    #当场景被激活，或者参数发生改变，更新图像
    @on_trait_change(['n_meridional','n_longitudinal'])
    #似乎监听scene.activated也可以实现在生成scene时计进入下面函数方法
    def update_plot(self):
        x, y, z, t = curve(self.n_meridional, self.n_longitudinal)

        if self.plot is None: # 如果plot未绘制则输出plot3d
            self.plot = self.scene.mlab.plot3d(x, y, z, t,
                                               tube_radius=0.025, colormap="Spectral")
        else: # 如果没有数据变化，将数据更新，重新赋值
            self.plot.mlab_source.set(
                x=x, y=y, z=z, scalars=t
            )

    #建立视图布局
    view = View(
        Item("scene",editor=SceneEditor(scene_class=MayaviScene),
             height=250,width=300,show_label=False),
        Group("_","n_meridional","n_longitudinal"),
        resizable=True
    )

model = MyModel()
model.configure_traits()