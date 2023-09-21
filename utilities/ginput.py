#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ginput class for Python

@author: Patrik Müller
@date:   12.12.2019
"""

from matplotlib.backend_bases import MouseButton as mb
import matplotlib.pyplot as plt
import numpy as np


class Ginput:
    """
        Interactive image plot for drawing points in a 2D-space

        Args:
            fig (plt.Figure): Figure containing the interactive plot
            ax (plt.Axes): Axes object which will be the interactive plot
            groups (int): Number of different point groups (1 <= groups <= 2)
            linestyle (str): Linestyle connecting two consecutive points
    """

    def __init__(self, fig, ax, groups=1, linestyle=''):
        if groups < 1 or groups > 2:
            ValueError('groups must be 1 or 2')
        self._points = []
        for _ in range(groups):
            self._points.append([])
        self._fig = fig
        self._ax = ax
        self._run = True
        self._colors = ['r', 'b']
        self._groups = [str(i) for i in range(groups)]
        ax.axis('square')
        ax.axis([0, 1, 0, 1])
        ax.grid('on')
        for i in range(groups):
            ax.plot(0, 0, linestyle=linestyle, color=self._colors[i],
                    label=self._groups[i], marker='o')
        if groups > 1:
            self._ax.legend()
        for i in range(groups):
            ax.lines[i].set_marker('')
        self._ax = ax
        self._cids = [None, None]

    def start(self):
        """ Start interactive plotting and connects all callbacks to figure."""
        text = 'Left click: Add point, Middle click: Cancel,' + \
               '\nRight click: Remove last point\n'
        if len(self._groups) > 1:
            text = text + 'Hold down SHIFT to switch between groups'

        # for i in range(len(self._groups)):
        #     self._ax.lines[i].set_linestyle('')
        self._ax.set_title(text)
        self._cids[0] = self._fig.canvas.mpl_connect('button_press_event',
                                                     self.on_press)
        self._cids[1] = self._fig.canvas.mpl_connect('close_event',
                                                     self.on_close)
        plt.draw()
        while self._run:
            plt.pause(0.1)
        self.stop()

    def stop(self):
        """ Stops execution of interactive plot and disconnects all callbacks
        to figure
        """
        self._run = False
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)

    def on_close(self, event):
        """ Callback for figure close event

        Args:
            event (obj): figure close event
        """
        self._run = False

    def update_plot(self):
        """ Update interactive plot """
        for i, d in enumerate(self._points):
            if len(d) > 0:
                points = np.array(d)
                x = points[:, 0]
                y = points[:, 1]
                self._ax.lines[i].set_data(x, y)
                self._ax.lines[i].set_marker('o')
            else:
                self._ax.lines[i].set_marker('None')

    def on_press(self, event):
        """ Callback for mouse clicks on figure. Adds valid points to
        list and displays them in plot.

        Args:
            event (obj): mouse click event
        """
        group = 0
        if event.key == 'shift' and len(self._groups) > 1:
            group = 1
        if event.button == mb.LEFT:
            if event.xdata is not None:
                x = event.xdata
                y = event.ydata
                point = [x, y]
                self._points[group].append(point)
        elif event.button == mb.RIGHT:
            if len(self._points[group]) > 0:
                self._points[group] = self._points[group][:-1]
        else:
            self._run = False
        self.update_plot()

    @property
    def points(self):
        """ Getter for registered points """
        return np.squeeze(np.array([np.squeeze(np.array(p)) for p in
                                    self._points]))


class Ginput_Image(Ginput):
    """
        Interactive image plot for drawing boundaries in an image

        Args:
            fig (plt.Figure): Figure containing the interactive plot
            ax (plt.Axes): Axes object which will be the interactive plot
            image (np.nparray): Image to display and draw boundaries in
    """

    def __init__(self, fig, ax, image):
        super().__init__(fig, ax, groups=1, linestyle='')
        self._ax.clear()
        self._ax.imshow(image, cmap='gray')
        self._ax.axis('off')
        self._image = image

    def update_plot(self):
        """ Updates interactive plot """
        self._ax.imshow(self._image, cmap='gray')

    def on_press(self, event):
        """ Callback for mouse clicks on figure. Adds valid points to
        list and marks them white in the image.

        Args:
            event (obj): mouse click event
        """
        if event.button == mb.LEFT:
            if event.xdata is not None:
                y = int(np.round(event.xdata))
                x = int(np.round(event.ydata))
                point = [x, y]
                if point in self._points[0]:
                    return
                self._image[x, y] = 1
                self._points[0].append(point)
        elif event.button == mb.RIGHT:
            if len(self._points) > 0:
                [x, y] = self._points[0][-1]
                self._image[x, y] = 0
                self._points[0] = self._points[0][:-1]
        else:
            self._run = False
        self.update_plot()

    @property
    def image(self):
        """ Getter for displayed image """
        return self._image


def test():
    """ Interactive test of Ginput classes """
    fig, ax = plt.subplots()
    ginp = Ginput(fig, ax, groups=1, linestyle='-')
    ginp.start()
    print(ginp.points)

    fig2, ax2 = plt.subplots()
    N = 15
    img = np.zeros((N, N))
    ginp2 = Ginput_Image(fig2, ax2, img)
    ginp2.start()
    print(ginp2.points)
    plt.figure()
    plt.imshow(ginp2.image)
    return


if __name__ == '__main__':
    test()
