import matplotlib
import mpl_toolkits
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class PageSlider(matplotlib.widgets.Slider):
    def __init__(self, ax, label, min_page = 0, max_page = 10, valinit=0, valfmt='%+1d', 
                 closedmin=True, closedmax=True,  
                 dragging=True, **kwargs):

        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = max_page - min_page + 1
        self.min_page = min_page

        super(PageSlider, self).__init__(ax, label, min_page, max_page, 
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(min_page, max_page+1):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i-min_page)/self.numpages, 0), 1./self.numpages, 1, 
                                transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i-min_page)/self.numpages+0.5/self.numpages, 0.5, str(i),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label='<',
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label='>',
                        color=self.facecolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.backward)
        self.button_forward.on_clicked(self.forward)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = self._int_val(self.val)
        if i >= self.valmax:
            i = self.valmax
        self._colorize(i - self.min_page)

    def _int_val(self, val):
        i = 0
        if val > 0:
            i = int(val + 0.5)
        if val < 0:
            i = int(val - 0.5)
        return i

    def on_changed(self, func):
        super(PageSlider, self).on_changed(lambda val: func(self._int_val(val)))

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def forward(self, event):
        current_i = 0
        if self.val > 0:
            current_i = int(self.val + 0.5)
        if self.val < 0:
            current_i = int(self.val - 0.5)
        i = current_i+1
        if (i < self.valmin) or (i > self.valmax):
            return
        self.set_val(i)
        self._colorize(i - self.min_page)

    def backward(self, event):
        current_i = 0
        if self.val > 0:
            current_i = int(self.val + 0.5)
        if self.val < 0:
            current_i = int(self.val - 0.5)
        i = current_i-1
        if (i < self.valmin) or (i > self.valmax):
            return
        self.set_val(i)
        self._colorize(i - self.min_page)
