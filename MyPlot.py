import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib import cm

def fig_ax_for_save(kwargs={}, name='', fontsize=20, height=10, width=10, ticks=True, axis_label=True):
    fig, ax= plt.subplots(subplot_kw=kwargs)
    if name:
        fig.suptitle(name, fontsize=fontsize)
        
    fig.set_figheight(height)
    fig.set_figwidth(width)
    if not axis_label:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, ax

def multi_cof_draw_img(name, pre, ans, cof, GridSize, a=1, levels=None):
    pre = pre.reshape(GridSize, GridSize)
    ans = ans.reshape(GridSize, GridSize)

    fig = plt.figure()
    fig.suptitle(name, fontsize=20)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    h = a / GridSize
    xx, yy = np.meshgrid(
        np.arange(h/2, a, h),
        np.arange(h/2, a, h)
    )

    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
    ax4 = fig.add_subplot(2, 2, 4, aspect="equal")
    
    im = ax1.imshow(cof, origin='lower')
    ax1.set_title(f'$Source$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax1)
    cbar.ax.tick_params(labelsize=10)

    ax2.set_title(f'$Prediction$', fontsize=20)
    surf_pre = ax2.plot_surface(xx, yy, pre, cmap=cm.Spectral_r,)
    cbar=plt.colorbar(surf_pre, shrink=0.85, ax=ax2)
    cbar.ax.tick_params(labelsize=10)

    diff = np.abs(pre - ans)
    ctf = ax3.contourf(xx, yy, diff, levels=50)
    ax3.set_title(f'$Difference$', fontsize=20)
    cbar=plt.colorbar(ctf, shrink=0.85, ax=ax3)
    cbar.ax.tick_params(labelsize=10)

    if levels is None:
        levels = np.linspace(ans.min(), ans.max(), 10)[2:-2] 
    ct1 = ax4.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5,  levels=levels)
    ct2 = ax4.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
    ax4.clabel(ct1, inline=False, fontsize=20)
    ax4.clabel(ct2, inline=False, fontsize=20)
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
    red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
    ax4.legend(handles=[blue_line, red_line], fontsize=20 )

    fig.tight_layout()
    return fig

def multi_water_draw_img(name, f, boundary, pre, ans, GridSize, a=1, levels=None):
    fig = plt.figure()
    fig.suptitle(name, fontsize=20)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    h = a / GridSize
    xx, yy = (
        np.arange(h/2, a, h),
        np.arange(h/2, a, h)
    )

    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
    ax4 = fig.add_subplot(2, 2, 4, aspect="equal")
    
    im = ax1.imshow(f, origin='lower')
    ax1.set_title(f'$Source$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax1)
    cbar.ax.tick_params(labelsize=10)

    ax2.set_title(f'$Prediction$', fontsize=20)
    surf_pre = ax2.plot_surface(xx, yy, pre, cmap=cm.Spectral_r,)
    cbar=plt.colorbar(surf_pre, shrink=0.85, ax=ax2)
    cbar.ax.tick_params(labelsize=10)

    im = ax3.imshow(boundary, origin='lower')
    ax3.set_title(f'$boundary$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax3)
    cbar.ax.tick_params(labelsize=10)

    if levels is None:
        levels = np.linspace(ans.min(), ans.max(), 10)[4:-1] 
    ct1 = ax4.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5,  levels=levels)
    ct2 = ax4.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
    ax4.clabel(ct1, inline=False, fontsize=20)
    ax4.clabel(ct2, inline=False, fontsize=20)
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
    red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
    ax4.legend(handles=[blue_line, red_line], fontsize=16 )

    fig.tight_layout()
    return fig

def multi_heat_draw_img(name, f, boundary, pre, ans, GridSize, a=0.1, levels=None):
    fig = plt.figure()
    fig.suptitle(name, fontsize=20)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    h = a / GridSize
    xx, yy = (
        np.arange(h/2, a, h),
        np.arange(h/2, a, h)
    )

    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax2 = fig.add_subplot(2, 2, 2, aspect="equal")
    ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
    ax4 = fig.add_subplot(2, 2, 4, aspect="equal")
    
    im = ax1.imshow(f, origin='lower')
    ax1.set_title(f'$Source$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax1)
    cbar.ax.tick_params(labelsize=10)

    ax2.set_title(f'$Prediction$', fontsize=20)
    ctf_pre = ax2.contourf(xx, yy, pre, cmap=cm.Spectral_r, levels=50)
    cbar=plt.colorbar(ctf_pre, shrink=0.85, ax=ax2)
    cbar.ax.tick_params(labelsize=10)

    im = ax3.imshow(boundary, origin='lower')
    ax3.set_title(f'$boundary$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax3)
    cbar.ax.tick_params(labelsize=10)

    if levels is None:
        levels = np.linspace(ans.min(), ans.max(), 10)[2:-2] 
    ct1 = ax4.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5,  levels=levels)
    ct2 = ax4.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
    ax4.clabel(ct1, inline=False, fontsize=20)
    ax4.clabel(ct2, inline=False, fontsize=20)
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
    red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
    ax4.legend(handles=[blue_line, red_line], fontsize=16 )

    fig.tight_layout()
    return fig

def multi_nonlinear_draw_img(name, f, mu, pre, ans, GridSize, a=1, levels=None):
    fig = plt.figure()
    fig.suptitle(f"{name}-{mu:.3e}", fontsize=20)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    h = a / GridSize
    xx, yy = (
        np.arange(h/2, a, h),
        np.arange(h/2, a, h)
    )

    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax2 = fig.add_subplot(2, 2, 2, aspect="equal")
    ax3 = fig.add_subplot(2, 2, 3, aspect="equal")
    ax4 = fig.add_subplot(2, 2, 4, aspect="equal")
    
    im = ax1.imshow(f, origin='lower')
    ax1.set_title(f'$Source$', fontsize=20)
    cbar=plt.colorbar(im, shrink=0.85, ax=ax1)
    cbar.ax.tick_params(labelsize=10)

    ax2.set_title(f'$Prediction$', fontsize=20)
    ctf_pre = ax2.contourf(xx, yy, pre, cmap=cm.Spectral_r,)
    cbar=plt.colorbar(ctf_pre, shrink=0.85, ax=ax2)
    cbar.ax.tick_params(labelsize=10)

    ax3.set_title(f'$Difference$', fontsize=20)
    ctf_diff = ax3.contourf(xx, yy, np.abs(pre - ans), cmap=cm.Spectral_r,)
    cbar=plt.colorbar(ctf_diff, shrink=0.85, ax=ax3)
    cbar.ax.tick_params(labelsize=10)

    if levels is None:
        levels = np.linspace(ans.min(), ans.max(), 10)[4:-1] 
    ct1 = ax4.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5,  levels=levels)
    ct2 = ax4.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
    ax4.clabel(ct1, inline=False, fontsize=20)
    ax4.clabel(ct2, inline=False, fontsize=20)
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
    red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
    ax4.legend(handles=[blue_line, red_line], fontsize=16 )

    fig.tight_layout()
    return fig

def save_img_force(path, f,):
    # Plot force function f
    fig, ax = fig_ax_for_save(ticks=False, axis_label=False)
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(f[::-1])
    # cbar=plt.colorbar(surf_pre, shrink=0.85, ax=ax)
    # cbar.ax.tick_params(labelsize=10)
    fig.savefig(f"{path/'force.png'}", bbox_inches='tight')
    plt.close(fig)

def save_surf(path, z, xx, yy, name='surf_pre'):
    # plot surfaces of pre and ans
    fig, ax = fig_ax_for_save({"projection": "3d"})
    surf_pre = ax.plot_surface(xx, yy, z, cmap=cm.Spectral_r,)
    cbar=plt.colorbar(surf_pre, shrink=0.85, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(f"{path/f'{name}.png'}", bbox_inches='tight')
    plt.close(fig)

def save_ctf(path, pre, ans, xx, yy):
    # plot contourf of difference between real answer and prediction
    fig, ax = fig_ax_for_save({}, ticks=False, axis_label=False)
    ax.set_aspect('equal', adjustable='box')
    ct = ax.contourf(xx, yy, np.abs(ans - pre), cmap=cm.Spectral_r, levels=50)
    cbar=plt.colorbar(ct, shrink=0.85, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(f"{path/'ctf_diff.png'}", bbox_inches='tight')
    plt.close(fig)

    # plot contourf of pre and ref
    fig, ax = fig_ax_for_save({}, ticks=False, axis_label=False)
    ax.set_aspect('equal', adjustable='box')
    ctf = ax.contourf(xx, yy, pre, alpha=1, cmap=cm.Spectral_r, levels=50)
    cbar=plt.colorbar(ctf, shrink=0.85, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(f"{path/'ctf_pre.png'}", bbox_inches='tight')
    plt.close(fig)

    fig, ax = fig_ax_for_save({}, ticks=False, axis_label=False)
    ax.set_aspect('equal', adjustable='box')
    ctf = ax.contourf(xx, yy, ans, alpha=1, cmap=cm.Spectral_r, levels=50)
    cbar=plt.colorbar(ctf, shrink=0.85, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(f"{path/'ctf_ref.png'}", bbox_inches='tight')
    plt.close(fig)

def save_contour(path, pre, ans, xx, yy, levels):
    # plot contour of prediction and real answer
    fig, ax = fig_ax_for_save({}, ticks=False, axis_label=False)
    ax.set_aspect('equal', adjustable='box')

    if levels is None:
        levels = np.linspace(ans.min(), ans.max(), 8)[1:-1]
    ct1 = ax.contour(xx, yy, pre, colors='r', linestyles='dashed', linewidths=1.5, levels=levels)
    ct2 = ax.contour(xx, yy, ans, colors='b', linestyles='solid', linewidths=2, levels=levels)
    ax.clabel(ct1, inline=False, fontsize=20)
    ax.clabel(ct2, inline=False, fontsize=20)
    blue_line = mlines.Line2D([], [], color='blue', markersize=20, label='ref')
    red_line = mlines.Line2D([], [], color='red', markersize=20, label='pre')
    ax.legend(handles=[blue_line, red_line], fontsize=16 )
    fig.savefig(f"{path/'ct.png'}", bbox_inches='tight')
    plt.close(fig)

def save_img(path, f, pre, ans, xx, yy, levels=None):
    p = Path(path)
    if not p.is_dir(): p.mkdir(parents=True)

    save_img_force(path, f)
    save_surf(path, pre, xx, yy, 'surf_pre')
    save_surf(path, ans, xx, yy, 'surf_ans')
    save_ctf(path, pre, ans, xx, yy)
    save_contour(path, pre, ans, xx, yy, levels=None)
    return 

