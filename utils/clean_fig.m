function clean_fig(fig_handle)

fig_axes = gca(fig_handle);
fig_axes.FontSize = 15;
fig_axes.Box = 'off';
end